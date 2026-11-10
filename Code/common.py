"""Helpers shared by notebooks 01-04, so the split, the label format and the metrics are
computed by one piece of code. Notebooks add this folder to sys.path (see their first cell).

Record format used everywhere (real and synthetic images):
    {"id", "path", "width", "height", "boxes": [[x0, y0, x1, y1]], "labels": [class index],
     "condition", "kind": "real" | "synthetic", ...}
"""
import json
import os
import random
import re
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

# Course rule: pick the device dynamically, move models and tensors to it explicitly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TABLE_COLUMNS = ["name", "role", "lookalike_of", "appearance", "cause_or_context",
                 "size_range", "how_to_generate", "source"]

# Short phrases for Grounding-DINO (filter in 02, zero-shot baseline in 04). The objects
# table has long descriptions; the open-vocabulary detector works better with short nouns.
DINO_PHRASE = {
    "fire_extinguisher": "fire extinguisher", "fake_extinguisher": "fake fire extinguisher",
    "vent_cover": "ventilation cover", "vent_no_cover": "open ventilation hole",
    "shelter_sign": "shelter sign", "wrong_sign": "wrong sign",
    "light_fixture": "ceiling light", "drinking_water": "water bottle",
    "gas_cylinder": "gas cylinder", "glass_item": "mirror",
}


def phrase(name):
    return DINO_PHRASE.get(name, name.replace("_", " "))


# ----------------------------------------------------------------------------- config / table
def load_objects(repo, cfg):
    """Read Data/objects_table.csv, drop cfg['exclude_classes'], validate it (PRD 3.2, 10)."""
    df = pd.read_csv(f"{repo}/Data/objects_table.csv").fillna("")
    missing = [c for c in TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"objects_table.csv is missing columns: {missing}")
    for col in TABLE_COLUMNS:
        if col != "lookalike_of" and (df[col] == "").any():
            raise ValueError(f"objects_table.csv: empty cells in column '{col}'")
    excluded = set(cfg.get("exclude_classes") or [])
    unknown = excluded - set(df["name"])
    if unknown:
        raise ValueError(f"exclude_classes names not in the table: {sorted(unknown)}")
    df = df[~df["name"].isin(excluded)].reset_index(drop=True)
    required = set(df.loc[df["role"] == "required", "name"])
    for _, row in df.iterrows():
        if row["role"] == "lookalike" and row["lookalike_of"] not in required:
            raise ValueError(f"Lookalike '{row['name']}' points to '{row['lookalike_of']}', which is not a "
                             "required class in the table. If you excluded it, exclude this lookalike too.")
        if row["role"] != "lookalike" and row["lookalike_of"] != "":
            raise ValueError(f"'{row['name']}' has lookalike_of but its role is {row['role']}")
    return df


def parse_size(text):
    """'40-160 px (estimate)' -> (40, 160)."""
    lo, hi = [int(x) for x in re.findall(r"\d+", text)[:2]]
    return lo, hi


def check_models(model_ids):
    """Fail early, with a clear message, if a model name in config.yaml does not exist (PRD 10)."""
    from huggingface_hub import model_info
    from huggingface_hub.utils import RepositoryNotFoundError
    for model_id in model_ids:
        try:
            model_info(model_id)
        except RepositoryNotFoundError:
            raise ValueError(f"Model {model_id} was not found on Hugging Face. Fix the name in config.yaml.")


# ----------------------------------------------------------------------------- real images
def load_real(root, classes):
    """Real images from Drive: real/images/ + real/annotations.json (COCO, with source_url and condition)."""
    ann_path = f"{root}/real/annotations.json"
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"{ann_path} not found. Put the labeled real images in Drive: real/images/ and "
                                "real/annotations.json (COCO format, category names = names in objects_table.csv).")
    coco = json.load(open(ann_path))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    label_of = {name: i for i, name in enumerate(classes)}
    skipped = sorted({n for n in cat_name.values() if n not in label_of})
    if skipped:
        print("Annotations of these categories are ignored (not in the class list):", skipped)
    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    records, unlabeled, no_file, no_source = [], [], [], []
    for im in sorted(coco["images"], key=lambda x: x["id"]):
        path = f"{root}/real/images/{im['file_name']}"
        w, h = im["width"], im["height"]
        boxes, labels = [], []
        for a in by_image.get(im["id"], []):
            name = cat_name[a["category_id"]]
            if name not in label_of:
                continue
            x, y, bw, bh = a["bbox"]
            box = [max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)]
            if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
                boxes.append(box)
                labels.append(label_of[name])
        if not boxes:
            unlabeled.append(im["file_name"])      # PRD 10: no labels -> flagged, not used
        elif not os.path.exists(path):
            no_file.append(im["file_name"])
        else:
            if not im.get("source_url"):
                no_source.append(im["file_name"])
            records.append({"id": f"real_{im['id']}", "path": path, "width": w, "height": h, "boxes": boxes,
                            "labels": labels, "condition": im.get("condition", "normal"),
                            "source_url": im.get("source_url", ""), "kind": "real"})
    for msg, names in [("without labels (not used)", unlabeled), ("listed but file missing in Drive", no_file),
                       ("without source_url (PRD 3.3 needs one per image)", no_source)]:
        if names:
            print(f"WARNING: {len(names)} real images {msg}: {names[:5]}{' ...' if len(names) > 5 else ''}")
    return records


def split_real(records, cfg, root=None):
    """Deterministic split (seed 42): 80% tune+val, 20% test; 10% of the 80% is the control (val) split."""
    recs = sorted(records, key=lambda r: r["id"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    n_test = round(len(recs) * cfg["split"]["test"])
    train, test = torch.utils.data.random_split(range(len(recs)), [len(recs) - n_test, n_test], generator=gen)
    train_idx = list(train.indices)                  # positions in recs
    n_val = round(len(train_idx) * cfg["split"]["val_from_train"])
    tune, val = torch.utils.data.random_split(range(len(train_idx)), [len(train_idx) - n_val, n_val], generator=gen)
    for i in test.indices:
        recs[i]["split"] = "test"
    for name, subset in [("tune", tune), ("val", val)]:
        for pos in subset.indices:                   # positions in train_idx
            recs[train_idx[pos]]["split"] = name
    if root:
        os.makedirs(f"{root}/real", exist_ok=True)
        json.dump({r["id"]: r["split"] for r in recs}, open(f"{root}/real/splits.json", "w"), indent=1)
    return recs


def _dhash(path, size=8):
    img = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    return (img[:, 1:] > img[:, :-1]).flatten()


def find_duplicates(records, max_distance=4):
    """Pairs of near-identical images that sit in different splits (PRD 10)."""
    hashes = {r["id"]: _dhash(r["path"]) for r in records}
    pairs = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a["split"] != b["split"] and int((hashes[a["id"]] != hashes[b["id"]]).sum()) <= max_distance:
                pairs.append((a["id"], a["split"], b["id"], b["split"]))
    return pairs


def split_table(records, classes):
    """Number of images / boxes per class in each split (PRD flow 1, step 6)."""
    rows = []
    for name in ["tune", "val", "test"]:
        part = [r for r in records if r["split"] == name]
        row = {"split": name, "images": len(part)}
        for ci, c in enumerate(classes):
            row[c] = sum(r["labels"].count(ci) for r in part)
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def check_test_requirements(records, table):
    """PRD 3.3: >=20 test images, >=10 with a lookalike, >=5 with a forbidden object. Warn only."""
    classes = table["name"].tolist()
    test = [r for r in records if r["split"] == "test"]
    look = {classes.index(n) for n in table.loc[table["role"] == "lookalike", "name"]}
    forb = {classes.index(n) for n in table.loc[table["role"] == "forbidden", "name"]}
    counts = {"test images (need 20)": (len(test), 20),
              "with a lookalike (need 10)": (sum(bool(look & set(r["labels"])) for r in test), 10),
              "with a forbidden object (need 5)": (sum(bool(forb & set(r["labels"])) for r in test), 5)}
    for label, (n, need) in counts.items():
        print(("OK      " if n >= need else "WARNING ") + f"{label}: {n}")


# ----------------------------------------------------------------------------- synthetic images
def synthetic_paths(root, item_id):
    return f"{root}/synthetic/images/{item_id}.png", f"{root}/synthetic/meta/{item_id}.json"


def item_done(root, item_id):
    """The meta file is written last, so it marks the item as finished (resume after a disconnect)."""
    return os.path.exists(synthetic_paths(root, item_id)[1])


def save_item(root, item_id, image, meta):
    img_path, meta_path = synthetic_paths(root, item_id)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    image.save(img_path)
    json.dump(meta, open(meta_path, "w"), indent=1)


def read_meta(root):
    """All meta dicts of the synthetic folder (one JSON per image)."""
    folder = f"{root}/synthetic/meta"
    if not os.path.isdir(folder):
        return []
    return [json.load(open(f"{folder}/{f}")) for f in sorted(os.listdir(folder)) if f.endswith(".json")]


def load_synthetic(root, classes, passed_only=True):
    """Synthetic images as records. With passed_only, the filter of notebook 02 must have run."""
    label_of = {name: i for i, name in enumerate(classes)}
    records = []
    for m in read_meta(root):
        if passed_only:
            if m.get("filter") is None:
                raise RuntimeError("Synthetic images have not been filtered yet. Run 02_validate first.")
            if not m["filter"]["passed"]:
                continue
        records.append({"id": m["id"], "path": f"{root}/synthetic/{m['file']}", "width": m["size"][0],
                        "height": m["size"][1], "boxes": [b["xyxy"] for b in m["boxes"]],
                        "labels": [label_of[b["name"]] for b in m["boxes"]], "condition": m["condition"],
                        "method": "A" if m["method"] == "add_objects" else "B",
                        "source_id": m.get("source_id"), "kind": "synthetic"})
    return records


# ----------------------------------------------------------------------------- experiment modes
MODES = {1: "real", 2: "real + A", 3: "real + B", 4: "real + A + B", 5: "real + classical"}
CLASSICAL_OPS = ["dark", "blur", "noise", "bright"]


def classical_aug(img, op, seed):
    """Augmentation without GenAI (mode 5). It never moves pixels, so boxes stay valid."""
    if op == "dark":
        return ImageEnhance.Brightness(img).enhance(0.35)
    if op == "bright":
        return ImageEnhance.Brightness(img).enhance(1.6)
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(3))
    noise = np.random.default_rng(seed).normal(0, 25, (img.height, img.width, 3))
    return Image.fromarray(np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype(np.uint8))


def balanced_sample(records, n, key, rng):
    """n records, taken round-robin over groups so every class/condition is represented equally."""
    groups = {}
    for r in records:
        groups.setdefault(key(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, i = [], 0
    while len(out) < n and i < max(len(g) for g in groups.values()):
        for k in sorted(groups):
            if i < len(groups[k]) and len(out) < n:
                out.append(groups[k][i])
        i += 1
    return out


def build_modes(tune, syn_a, syn_b, n, seed, image_size):
    """Training records for the 5 modes. Modes 2-5 get n extra images each (PRD 3.9 fairness rule):
    mode 4 = n/2 from A + n/2 from B; mode 5 = classical augmentation of the *source images of mode 3*."""
    if n % 2:
        raise ValueError("training.synthetic_per_mode must be even (mode 4 takes half from A, half from B)")
    rng = random.Random(seed)
    if len(syn_a) < n or len(syn_b) < n:
        raise ValueError(f"Need {n} filtered images of each method, have A={len(syn_a)}, B={len(syn_b)}. "
                         "Generate more or lower training.synthetic_per_mode.")
    a = balanced_sample(syn_a, n, lambda r: r["labels"][0], rng)
    b = balanced_sample(syn_b, n, lambda r: r["condition"], rng)
    by_id = {r["id"]: r for r in tune}
    classical = []
    for i, rec in enumerate(b):
        if rec["source_id"] not in by_id:
            raise KeyError(f"{rec['id']} was made from {rec['source_id']}, which is not in the tune split anymore. "
                           "The real annotations changed after generation; regenerate method B.")
        src = by_id[rec["source_id"]]
        sx, sy = image_size / src["width"], image_size / src["height"]
        classical.append({**src, "id": f"{src['id']}_cl{i}", "aug": CLASSICAL_OPS[i % len(CLASSICAL_OPS)],
                          "aug_seed": seed + i, "resize_to": image_size, "width": image_size, "height": image_size,
                          "boxes": [[x0 * sx, y0 * sy, x1 * sx, y1 * sy] for x0, y0, x1, y1 in src["boxes"]],
                          "kind": "classical"})
    return {1: list(tune), 2: list(tune) + a, 3: list(tune) + b, 4: list(tune) + a[:n // 2] + b[:n // 2],
            5: list(tune) + classical}


def cache_local(records, cache_dir="/content/data_cache"):
    """Copy images from Drive to the Colab disk once (course pattern: faster reads during training)."""
    out = []
    for r in records:
        local = os.path.join(cache_dir, os.path.relpath(r["path"], "/content/drive"))
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copy(r["path"], local)
        out.append({**r, "path": local})
    return out


# ----------------------------------------------------------------------------- detection models
class DetectionDataset(torch.utils.data.Dataset):
    """Records -> what AutoModelForObjectDetection wants: pixel_values + labels (COCO format via the processor)."""

    def __init__(self, records, processor):
        self.records, self.processor = records, processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["path"]).convert("RGB")
        if r.get("resize_to"):                       # classical-augmentation records live at the generation size
            s = r["resize_to"]
            img = img.resize((s, s))
        if r.get("aug"):
            img = classical_aug(img, r["aug"], r["aug_seed"])
        anns = [{"id": j, "image_id": i, "category_id": int(label), "area": float((x1 - x0) * (y1 - y0)),
                 "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "iscrowd": 0}
                for j, ((x0, y0, x1, y1), label) in enumerate(zip(r["boxes"], r["labels"]))]
        enc = self.processor(images=img, annotations={"image_id": i, "annotations": anns}, return_tensors="pt")
        item = {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}
        if "pixel_mask" in enc:
            item["pixel_mask"] = enc["pixel_mask"][0]
        return item


def collate_fn(batch):
    out = {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": [b["labels"] for b in batch]}
    if "pixel_mask" in batch[0]:
        out["pixel_mask"] = torch.stack([b["pixel_mask"] for b in batch])
    return out


def freeze_backbone(model):
    """Course rule: freeze the pretrained CNN backbone, train the transformer encoder/decoder and the heads."""
    backbone = getattr(getattr(model, "model", None), "backbone", None)
    if backbone is None:
        print("No separate backbone found, all weights stay trainable.")
    else:
        backbone.requires_grad_(False)
    return model


def load_detector(model_id, classes, size, freeze=True):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    id2label = dict(enumerate(classes))
    label2id = {c: i for i, c in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(model_id, size={"height": size, "width": size})
    model = AutoModelForObjectDetection.from_pretrained(model_id, num_labels=len(classes), id2label=id2label,
                                                        label2id=label2id, ignore_mismatched_sizes=True)
    return (freeze_backbone(model) if freeze else model), processor


def predict(model, processor, records, max_dets=100):
    """Raw detections for every record (all scores kept; thresholds are applied later)."""
    model.to(device).eval()
    preds = []
    for r in records:
        img = Image.open(r["path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=[(img.height, img.width)])[0]
        order = res["scores"].detach().cpu().numpy().argsort()[::-1][:max_dets]
        preds.append({"boxes": res["boxes"].detach().cpu().numpy()[order].reshape(-1, 4),
                      "scores": res["scores"].detach().cpu().numpy()[order],
                      "labels": res["labels"].detach().cpu().numpy()[order]})
    return preds


def predict_zero_shot(model, processor, records, classes, box_threshold=0.1):
    """Grounding-DINO with the class phrases as the text prompt (no training). Same output format as predict()."""
    model.to(device).eval()
    text = ". ".join(phrase(c) for c in classes) + "."
    by_phrase = {phrase(c): i for i, c in enumerate(classes)}
    preds = []
    for r in records:
        img = Image.open(r["path"]).convert("RGB")
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=box_threshold,
            target_sizes=[(img.height, img.width)])[0]
        names = res.get("text_labels", res.get("labels"))
        boxes, scores, labels = [], [], []
        for box, score, name in zip(res["boxes"].detach().cpu().numpy(), res["scores"].detach().cpu().numpy(), names):
            name = str(name).strip().lower()
            label = by_phrase.get(name)
            if label is None:                         # merged phrase: take the longest class phrase inside it
                inside = [p for p in by_phrase if p in name]
                label = by_phrase[max(inside, key=len)] if inside else None
            if label is not None:
                boxes.append(box)
                scores.append(score)
                labels.append(label)
        preds.append({"boxes": np.array(boxes, dtype=float).reshape(-1, 4), "scores": np.array(scores, dtype=float),
                      "labels": np.array(labels, dtype=int)})
    return preds


# ----------------------------------------------------------------------------- metrics
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(det, gt_boxes, gt_labels, thr, iou_thr=0.5):
    """Greedy one-to-one matching of detections (score >= thr) to ground truth of the same class.
    Returns ({gt index: det index}, [unmatched det indices])."""
    keep = np.where(det["scores"] >= thr)[0]
    order = keep[np.argsort(-det["scores"][keep])]
    matched = {}
    for d in order:
        best, best_iou = None, iou_thr
        for g in range(len(gt_boxes)):
            if g not in matched and gt_labels[g] == det["labels"][d]:
                v = iou(det["boxes"][d], gt_boxes[g])
                if v >= best_iou:
                    best, best_iou = g, v
        if best is not None:
            matched[best] = d
    used = set(matched.values())
    return matched, [d for d in order if d not in used]


def map50(preds, recs):
    """mAP@50 over the classes that have ground truth (torchmetrics)."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5])
    metric.update(
        [{"boxes": torch.tensor(p["boxes"]).float().reshape(-1, 4), "scores": torch.tensor(p["scores"]).float(),
          "labels": torch.tensor(p["labels"]).long()} for p in preds],
        [{"boxes": torch.tensor(r["boxes"]).float().reshape(-1, 4), "labels": torch.tensor(r["labels"]).long()}
         for r in recs])
    value = float(metric.compute()["map_50"])
    return float("nan") if value < 0 else value


def per_class_f1(preds, recs, thr, n_classes):
    """F1 per class at IoU 0.5 and confidence thr. NaN for classes with no ground truth."""
    tp, fp, fn = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    for p, r in zip(preds, recs):
        matched, unmatched = match(p, r["boxes"], r["labels"], thr)
        for g, label in enumerate(r["labels"]):
            (tp if g in matched else fn)[label] += 1
        for d in unmatched:
            fp[p["labels"][d]] += 1
    gt = tp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    return np.where(gt > 0, f1, np.nan)


def pick_threshold(preds, recs, n_classes, grid=np.arange(0.05, 0.96, 0.05)):
    """Confidence threshold with the best macro F1 on the control split (PRD 3.12)."""
    scores = [np.nanmean(per_class_f1(preds, recs, t, n_classes)) if len(recs) else 0.0 for t in grid]
    return float(grid[int(np.nanargmax(scores))])


def is_small(box, width, height, image_size, small_px):
    """COCO 'small': area below 32x32, measured at the generation size (images differ in resolution)."""
    scale = image_size / max(width, height)
    return (box[2] - box[0]) * (box[3] - box[1]) * scale * scale < small_px ** 2


def verdict(labels, required_ids, forbidden_ids):
    """PRD 3.12: valid only if every required object is present and no forbidden object is."""
    labels = set(labels)
    return int(all(r in labels for r in required_ids) and not any(f in labels for f in forbidden_ids))


def run_metrics(preds, recs, thr, table, cfg):
    """All per-run numbers of PRD 3.9 and 3.12 for one model/mode on one set of real images."""
    classes = table["name"].tolist()
    idx = {c: i for i, c in enumerate(classes)}
    f1 = per_class_f1(preds, recs, thr, len(classes))
    small_hit = small_all = look_total = look_false = forb_total = forb_missed = 0
    look_of = {idx[r["name"]]: idx[r["lookalike_of"]] for _, r in table.iterrows() if r["role"] == "lookalike"}
    forb = {idx[n] for n in table.loc[table["role"] == "forbidden", "name"]}
    req = [idx[n] for n in table.loc[table["role"] == "required", "name"]]
    y_true, y_pred = [], []
    for p, r in zip(preds, recs):
        matched, _ = match(p, r["boxes"], r["labels"], thr)
        for g, (box, label) in enumerate(zip(r["boxes"], r["labels"])):
            if is_small(box, r["width"], r["height"], cfg["generation"]["image_size"], cfg["small_object_px"]):
                small_all += 1
                small_hit += g in matched
            if label in look_of:      # a lookalike counts as falsely approved if the *genuine* class fires on it
                look_total += 1
                look_false += any(p["scores"][d] >= thr and p["labels"][d] == look_of[label] and iou(p["boxes"][d], box) >= 0.5
                                  for d in range(len(p["scores"])))
            if label in forb:
                forb_total += 1
                forb_missed += g not in matched
        y_true.append(verdict(r["labels"], req, forb))
        y_pred.append(verdict(p["labels"][p["scores"] >= thr], req, forb))
    rate = lambda a, b: a / b if b else float("nan")
    wrong_approve = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 1)
    return {"threshold": thr, "map50": map50(preds, recs), "macro_f1": float(np.nanmean(f1)),
            "f1_per_class": dict(zip(classes, [None if np.isnan(x) else float(x) for x in f1])),
            "small_recall": rate(small_hit, small_all), "n_small": small_all,
            "lookalike_false_approval": rate(look_false, look_total), "n_lookalike": look_total,
            "forbidden_miss": rate(forb_missed, forb_total), "n_forbidden": forb_total,
            "verdict_false_approval": rate(wrong_approve, y_true.count(0)),
            "y_true": y_true, "y_pred": y_pred}


def confusion(preds, recs, thr, n_classes):
    """Rows: true class (last row = no ground truth, a false alarm). Columns: detected class (last column = missed)."""
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for p, r in zip(preds, recs):
        keep = np.where(p["scores"] >= thr)[0]
        used = set()
        for box, label in zip(r["boxes"], r["labels"]):
            best, best_iou = None, 0.5
            for d in keep:
                v = iou(p["boxes"][d], box)
                if d not in used and v >= best_iou:
                    best, best_iou = d, v
            if best is None:
                m[label, n_classes] += 1
            else:
                used.add(best)
                m[label, p["labels"][best]] += 1
        for d in keep:
            if d not in used:
                m[n_classes, p["labels"][d]] += 1
    return m
