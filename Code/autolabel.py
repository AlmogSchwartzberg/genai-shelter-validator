"""Auto-label synthetic images: boxes -> verified YOLO labels (PRD 3.4, 3.5, 3.7 check 5).

Input : {drive_root}/synthetic/metadata.jsonl, one JSON object per image (written by 01_generate):
    {"image": "synthetic/images/000123.jpg",   # path relative to drive_root; keep images in synthetic/images/
     "method": "add_objects" | "scene_edits",  # names from config.yaml generation.methods
     "edit_type": null | "add_people" | ...,   # scene_edits only (config generation.scene_edits)
     "source": "real/abc.jpg",                 # scene_edits only, the real image that was edited
     "boxes": [{"class": "fire_extinguisher", "xyxy": [x0, y0, x1, y1]}],  # pixels in the saved image
     ... prompt, seed, model, attributes (PRD 9) are ignored here}
Output: {drive_root}/synthetic/labels/<image stem>.txt   YOLO lines, only for images that pass.
        With images in synthetic/images/ this is the images/ + labels/ layout YOLO training reads.
        A rejected image has no label file: train only on images that have one (YOLO treats a missing label as background)
        {drive_root}/synthetic/autolabel.jsonl           one line per image: status ok|rejected, reason, boxes
Class id = row order of Data/objects_table.csv (load_classes). Rerun resumes after the last image written.
"""
import csv
import json
from pathlib import Path

import numpy as np
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


def map_boxes(boxes, scale, dx, dy):
    """Boxes after the original image is resized by `scale` and pasted at (dx, dy) on a bigger canvas
    (room_size outpainting, PRD 3.5)."""
    return [[x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy] for x0, y0, x1, y1 in boxes]


def to_yolo(class_id, xyxy, w, h):
    x0, y0, x1, y1 = xyxy
    return f"{class_id} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def box_changed(src, out, xyxy):
    """Mean absolute pixel change (0..1) inside the box between two same-size images."""
    x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
    a = np.asarray(src.convert("RGB"), dtype=np.float32)[y0:y1, x0:x1]
    b = np.asarray(out.convert("RGB"), dtype=np.float32)[y0:y1, x0:x1]
    return float(np.abs(a - b).mean() / 255)


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
    """-> (status, reason, boxes with iou/final xyxy). Rejects the whole image if any box fails."""
    al, gen = cfg["autolabel"], cfg["generation"]
    img = Image.open(root / rec["image"])
    w, h = img.size
    src = None
    # scene_edits keep the real image's boxes: verify by pixels when the canvas is unchanged
    if rec["method"] == "scene_edits" and rec.get("edit_type") != "room_size":
        src = Image.open(root / rec["source"])
        if src.size != img.size:
            return "rejected", "source and edited image differ in size", []
    out = []
    for b in rec["boxes"]:
        cls, box = b["class"], [float(v) for v in b["xyxy"]]
        if cls not in names:
            raise ValueError(f"class {cls!r} is not in objects_table.csv")
        box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return "rejected", f"{cls}: box is empty or outside the image", out
        if src is not None:
            ch = box_changed(src, img, box)
            if ch > gen["box_change_threshold"]:
                return "rejected", f"{cls}: pixels inside the box changed by {ch:.2f}", out
            out.append({"class": cls, "xyxy": box})
            continue
        # a lookalike is checked with its own phrase if set, else with the object it imitates: the detector
        # confirms "an object of this kind sits here", not that the defect is right (that stays a manual check)
        phrase = al["prompts"].get(cls) or al["prompts"][parents[cls]]
        dets = detect(img, phrase)
        best = max(dets, key=lambda d: iou(box, d), default=None)
        score = iou(box, best) if best else 0.0
        if score < al["min_iou"]:
            return "rejected", f"{cls}: detector IoU {score:.2f} < {al['min_iou']}", out
        out.append({"class": cls, "xyxy": best if al["refine"] else box, "iou": round(score, 3)})
    return "ok", None, out


def run(cfg, root, names, parents, detect):
    """Label every image in metadata.jsonl that autolabel.jsonl does not list yet. -> counts per status."""
    root = Path(root)
    syn = root / "synthetic"
    (syn / "labels").mkdir(parents=True, exist_ok=True)
    done_file = syn / "autolabel.jsonl"
    done = {json.loads(l)["image"] for l in done_file.open()} if done_file.exists() else set()
    counts = {"ok": 0, "rejected": 0}
    with done_file.open("a") as f:
        for line in open(syn / "metadata.jsonl"):
            rec = json.loads(line)
            if rec["image"] in done:
                continue
            status, reason, boxes = label_image(rec, root, names, parents, cfg, detect)
            if status == "ok":
                w, h = Image.open(root / rec["image"]).size
                lines = [to_yolo(names.index(b["class"]), b["xyxy"], w, h) for b in boxes]
                (syn / "labels" / (Path(rec["image"]).stem + ".txt")).write_text("".join(l + "\n" for l in lines))
            f.write(json.dumps({"image": rec["image"], "status": status, "reason": reason, "boxes": boxes}) + "\n")
            f.flush()
            counts[status] += 1
    return counts
