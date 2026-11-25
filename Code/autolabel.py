"""Auto-label synthetic images: verify generated boxes, write YOLO labels (PRD 3.4, 3.7 check 5).

Input : {drive_root}/synthetic/metadata.jsonl, written by 01_generate (Code/synth.py, README "Synthetic data format").
        Used fields per line: "image" (path relative to drive_root) and
        "boxes": [{"class": name from objects_table.csv, "xyxy": [x0, y0, x1, y1] pixels of the saved image,
                   "origin": "synthetic" | "source"}]
        "synthetic" = the painted object (add_objects). "source" = a box from the real photo, kept and protected
        (shifted/scaled for room_size). Only synthetic boxes can be wrong, so only they go to the detector;
        source boxes are kept as they are (clipped to the image). A missing origin counts as synthetic.
Output: {drive_root}/synthetic/labels/<image stem>.txt   YOLO lines, only for images that pass
        {drive_root}/synthetic/autolabel.jsonl           one line per image: status ok|rejected, reason, boxes
        With images in synthetic/images/ this is the images/ + labels/ layout YOLO training reads. A rejected image
        has no label file: train only on images that have one (YOLO treats a missing label as background).
Class id = row order of Data/objects_table.csv (load_classes). Rerun resumes: an image is skipped only if its
journal line says rejected, or says ok and its label file exists.
"""
import csv
import json
import math
import os
from pathlib import Path

from PIL import Image


def load_classes(csv_path):
    """-> (names in row order, {lookalike name: the object it imitates})."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf8")))
    return [r["name"] for r in rows], {r["name"]: r["lookalike_of"] for r in rows if r["lookalike_of"]}


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clip_box(xyxy, w, h):
    """Box clipped to the image, or None if a coordinate is not finite or less than 2 px is left."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        return None
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    return [x0, y0, x1, y1] if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def to_yolo(class_id, xyxy, w, h):
    x0, y0, x1, y1 = xyxy
    return f"{class_id} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def read_jsonl(path):
    """Lines as dicts. A half-written last line (disconnect) is ignored; a bad line anywhere else is an error."""
    if not path.exists():
        return []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    out = []
    for i, l in enumerate(lines):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise
    return out


def grounding_dino(cfg):
    """Zero-shot detector as detect(image, phrase) -> [xyxy, ...]. Needs transformers and a GPU for speed."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    al = cfg["autolabel"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(al["detector"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(al["detector"]).to(device).eval()

    def detect(img, phrase):
        inputs = proc(images=img.convert("RGB"), text=phrase.lower().rstrip(".") + ".", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # positional args: the keyword for the box threshold differs between transformers versions
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, al["box_threshold"], al["text_threshold"], [img.size[::-1]])[0]
        return res["boxes"].detach().cpu().tolist()

    return detect


def label_image(rec, root, names, parents, cfg, detect):
    """-> (status, reason, boxes, (w, h)). Rejects the whole image if any box is invalid or a synthetic box fails."""
    al = cfg["autolabel"]
    img = Image.open(root / rec["image"])
    w, h = img.size
    if not rec["boxes"]:
        return "rejected", "no boxes", [], (w, h)
    out = []
    for b in rec["boxes"]:
        cls = b["class"]
        if cls not in names:
            raise ValueError(f"class {cls!r} is not in objects_table.csv")
        box = clip_box(b["xyxy"], w, h)
        if box is None:
            return "rejected", f"{cls}: box is not finite, empty or outside the image", out, (w, h)
        if b.get("origin", "synthetic") == "source":
            out.append({"class": cls, "xyxy": box})
            continue
        # a lookalike is checked with its own phrase if set, else with the object it imitates: the detector
        # confirms "an object of this kind sits here", not that the defect is right (that stays a manual check)
        phrase = al["prompts"].get(cls) or al["prompts"][parents[cls]]
        dets = [d for d in (clip_box(d, w, h) for d in detect(img, phrase)) if d]  # compare inside the image
        best = max(dets, key=lambda d: iou(box, d), default=None)
        score = iou(box, best) if best else 0.0
        if score < al["min_iou"]:
            return "rejected", f"{cls}: detector IoU {score:.2f} < {al['min_iou']}", out, (w, h)
        out.append({"class": cls, "xyxy": best if al["refine"] else box, "iou": round(score, 3)})
    return "ok", None, out, (w, h)


def run(cfg, root, names, parents, detect):
    """Label every image in metadata.jsonl that autolabel.jsonl does not settle yet. -> counts per status."""
    root = Path(root)
    syn = root / "synthetic"
    (syn / "labels").mkdir(parents=True, exist_ok=True)
    journal = syn / "autolabel.jsonl"
    # keep one line per image; drop an "ok" whose label file is gone and a half-written last line, so they are redone
    done = {e["image"]: e for e in read_jsonl(journal)
            if e["status"] == "rejected" or (syn / "labels" / (Path(e["image"]).stem + ".txt")).exists()}
    tmp = journal.with_name(journal.name + ".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in done.values()))
    os.replace(tmp, journal)
    counts = {"ok": 0, "rejected": 0}
    with journal.open("a") as f:
        for rec in read_jsonl(syn / "metadata.jsonl"):
            if rec["image"] in done:  # also skips a duplicate row of this run
                continue
            status, reason, boxes, (w, h) = label_image(rec, root, names, parents, cfg, detect)
            if status == "ok":
                lines = [to_yolo(names.index(b["class"]), b["xyxy"], w, h) for b in boxes]
                (syn / "labels" / (Path(rec["image"]).stem + ".txt")).write_text("".join(l + "\n" for l in lines))
            entry = {"image": rec["image"], "status": status, "reason": reason, "boxes": boxes}
            f.write(json.dumps(entry) + "\n")  # label first, journal second: a crash in between just redoes the image
            f.flush()
            done[rec["image"]] = entry
            counts[status] += 1
    return counts
