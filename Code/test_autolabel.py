"""Run: pytest Code/test_autolabel.py  (no GPU, no model: the detector is a stub)."""
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

import autolabel as al

REPO = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(REPO / "config.yaml"))
NAMES, PARENTS = al.load_classes(REPO / "Data/objects_table.csv")


def make_root(tmp_path, recs, n_images=6):
    syn = tmp_path / "synthetic"
    (syn / "images").mkdir(parents=True)
    for i in range(n_images):
        Image.new("RGB", (100, 100), "gray").save(syn / f"images/{i}.png")
    (syn / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return tmp_path


def rec(i, *boxes):
    return {"image": f"synthetic/images/{i}.png", "method": "add_objects", "boxes": list(boxes)}


def box(cls, xyxy, origin="synthetic"):
    return {"class": cls, "xyxy": xyxy, "origin": origin}


def stub(seen=None, boxes=((12, 12, 52, 52),)):
    def detect(img, phrase):
        if seen is not None:
            seen.append(phrase)
        return [] if phrase == "gas cylinder" else [list(b) for b in boxes]
    return detect


def test_classes_and_yolo():
    assert NAMES[0] == "fire_extinguisher" and PARENTS["fake_extinguisher"] == "fire_extinguisher"
    assert al.to_yolo(2, [10, 20, 30, 60], 100, 200) == "2 0.200000 0.200000 0.200000 0.200000"
    assert al.clip_box([-5, 0, 150, 50], 100, 100) == [0, 0, 100, 50]
    assert al.clip_box([float("nan"), 0, 50, 50], 100, 100) is None
    assert al.clip_box([150, 0, 200, 50], 100, 100) is None  # outside the image


def test_only_synthetic_boxes_are_verified(tmp_path):
    recs = [
        rec(0, box("fire_extinguisher", [10, 10, 50, 50])),  # detector agrees
        rec(1, box("fake_extinguisher", [10, 10, 50, 50])),  # lookalike is checked with its parent's phrase
        rec(2, box("gas_cylinder", [10, 10, 50, 50])),  # detector finds nothing
        # a real, hand-drawn gas cylinder box the detector would miss must not reject the image
        rec(3, box("gas_cylinder", [60, 60, 90, 90], "source"), box("fire_extinguisher", [10, 10, 50, 50])),
        rec(4, box("gas_cylinder", [60, 60, 90, 90], "source")),  # scene edit: nothing to verify
        rec(5, box("glass_item", [float("nan"), 0, 50, 50])),
    ]
    root = make_root(tmp_path, recs)
    seen = []
    assert al.run(CFG, root, NAMES, PARENTS, stub(seen)) == {"ok": 4, "rejected": 2}
    out = {e["image"][-5]: e for e in map(json.loads, open(root / "synthetic/autolabel.jsonl"))}
    assert [out[str(i)]["status"] for i in range(6)] == ["ok", "ok", "rejected", "ok", "ok", "rejected"]
    assert "gas_cylinder: detector IoU 0.00" in out["2"]["reason"] and "not finite" in out["5"]["reason"]
    assert seen == ["fire extinguisher", "fire extinguisher", "gas cylinder", "fire extinguisher"]
    lab = root / "synthetic/labels"
    assert (lab / "0.txt").read_text().startswith("0 0.300000 0.300000 0.400000 0.400000")
    assert (lab / "1.txt").read_text().startswith("1 ")  # the lookalike keeps its own class id
    assert len((lab / "3.txt").read_text().splitlines()) == 2 and not (lab / "2.txt").exists()


def test_refine_clips_to_the_image(tmp_path):
    cfg = {**CFG, "autolabel": {**CFG["autolabel"], "refine": True}}
    root = make_root(tmp_path, [rec(0, box("fire_extinguisher", [60, 60, 98, 98]))])
    assert al.run(cfg, root, NAMES, PARENTS, stub(boxes=[(58, 58, 130, 140)])) == {"ok": 1, "rejected": 0}
    vals = [float(v) for v in (root / "synthetic/labels/0.txt").read_text().split()[1:]]
    assert all(0 <= v <= 1 for v in vals) and vals[0] + vals[2] / 2 <= 1.0


def test_resume(tmp_path):
    root = make_root(tmp_path, [rec(0, box("fire_extinguisher", [10, 10, 50, 50])),
                                rec(0, box("fire_extinguisher", [10, 10, 50, 50])),  # duplicate row
                                rec(1, box("fire_extinguisher", [10, 10, 50, 50]))])
    assert al.run(CFG, root, NAMES, PARENTS, stub()) == {"ok": 2, "rejected": 0}
    assert al.run(CFG, root, NAMES, PARENTS, stub()) == {"ok": 0, "rejected": 0}
    syn = root / "synthetic"
    assert len(open(syn / "autolabel.jsonl").readlines()) == 2
    # the label file of image 0 is lost, and a disconnect cut the journal's last line in half
    (syn / "labels/0.txt").unlink()
    lines = open(syn / "autolabel.jsonl").read().splitlines()
    (syn / "autolabel.jsonl").write_text(lines[0] + "\n" + lines[1][:20])
    (syn / "metadata.jsonl").write_text(open(syn / "metadata.jsonl").read() + '{"image": "synthetic/images/2.p')
    assert al.run(CFG, root, NAMES, PARENTS, stub()) == {"ok": 2, "rejected": 0}
    assert (syn / "labels/0.txt").exists() and (syn / "labels/1.txt").exists()
    assert len(open(syn / "autolabel.jsonl").readlines()) == 2


def test_unknown_class_is_an_error(tmp_path):
    root = make_root(tmp_path, [rec(0, box("tv", [0, 0, 9, 9]))])
    with pytest.raises(ValueError):
        al.run(CFG, root, NAMES, PARENTS, stub())


def test_prelabel(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    for n in ("b.jpg", "a.jpg"):
        Image.new("RGB", (100, 80), "gray").save(real / n)
    (real / "annotations.json").write_text("{}")  # not a photo, ignored

    def detect(img, phrase):
        return [[10, 10, 50, 50], [float("nan"), 0, 5, 5]] if phrase == "gas cylinder" else []

    assert al.prelabel(CFG, tmp_path, NAMES, detect) == {"images": 2, "boxes": 2}
    coco = json.load(open(real / "prelabels.json"))
    assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "b.jpg"]
    gas = NAMES.index("gas_cylinder") + 1
    assert [(a["image_id"], a["category_id"], a["bbox"]) for a in coco["annotations"]] == [(1, gas, [10, 10, 40, 40]), (2, gas, [10, 10, 40, 40])]
    assert len(coco["categories"]) == len(NAMES) and coco["categories"][gas - 1]["name"] == "gas_cylinder"
    assert not any(c["name"] == "fake_extinguisher" and c["id"] in {a["category_id"] for a in coco["annotations"]} for c in coco["categories"])


def test_roboflow_upload(tmp_path, monkeypatch):
    import sys, types
    real = tmp_path / "real"
    real.mkdir()
    Image.new("RGB", (10, 10)).save(real / "a.jpg")
    (real / "prelabels.json").write_text("{}")
    calls = []

    class WS:
        def project(self, name):
            if name == "missing":
                raise RuntimeError("404")

        def upload_dataset(self, stage, project, **kw):
            calls.append((sorted(p.name for p in Path(stage).iterdir()), project, kw))

    fake = types.ModuleType("roboflow")
    fake.Roboflow = lambda api_key: types.SimpleNamespace(workspace=lambda name: WS())
    monkeypatch.setitem(sys.modules, "roboflow", fake)
    cfg = {"roboflow": {"workspace": "w", "project": "p"}}
    al.roboflow_upload(cfg, tmp_path, "key")
    assert calls == [(["_annotations.coco.json", "a.jpg"], "p", {"is_prediction": True, "batch_name": "prelabels"})]
    with pytest.raises(RuntimeError, match="Create it in the Roboflow UI"):
        al.roboflow_upload({"roboflow": {"workspace": "w", "project": "missing"}}, tmp_path, "key")
    assert len(calls) == 1  # nothing is created or uploaded when the project does not exist
