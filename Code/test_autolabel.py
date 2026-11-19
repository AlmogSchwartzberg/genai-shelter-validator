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


def make_root(tmp_path, recs):
    syn = tmp_path / "synthetic"
    syn.mkdir()
    (tmp_path / "real").mkdir()
    Image.new("RGB", (100, 100), "gray").save(tmp_path / "real/src.png")
    for i in range(4):
        Image.new("RGB", (100, 100), "gray").save(syn / f"{i}.png")
    Image.new("RGB", (100, 100), "white").save(syn / "changed.png")
    (syn / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return tmp_path


def box(cls, xyxy):
    return {"class": cls, "xyxy": xyxy}


def test_classes_and_yolo():
    assert NAMES[0] == "fire_extinguisher" and PARENTS["fake_extinguisher"] == "fire_extinguisher"
    assert al.to_yolo(2, [10, 20, 30, 60], 100, 200) == "2 0.200000 0.200000 0.200000 0.200000"
    assert al.map_boxes([[10, 10, 20, 20]], 0.5, 5, 5) == [[10, 10, 15, 15]]


def test_run(tmp_path):
    A = {"method": "add_objects", "edit_type": None}
    B = {"method": "scene_edits", "edit_type": "add_people", "source": "real/src.png"}
    recs = [
        {"image": "synthetic/0.png", **A, "boxes": [box("fire_extinguisher", [10, 10, 50, 50])]},  # detector agrees
        {"image": "synthetic/1.png", **A, "boxes": [box("fake_extinguisher", [10, 10, 50, 50])]},  # lookalike, parent phrase
        {"image": "synthetic/3.png", **A, "boxes": [box("gas_cylinder", [10, 10, 50, 50])]},  # detector finds nothing
        {"image": "synthetic/2.png", **B, "boxes": [box("glass_item", [0, 0, 10, 10])]},  # untouched pixels
        {"image": "synthetic/changed.png", **B, "boxes": [box("glass_item", [0, 0, 10, 10])]},  # object covered
    ]
    root = make_root(tmp_path, recs)
    seen = []

    def detect(img, phrase):
        seen.append(phrase)
        return [] if phrase == "gas cylinder" else [[12, 12, 52, 52]]

    assert al.run(CFG, root, NAMES, PARENTS, detect) == {"ok": 3, "rejected": 2}
    out = [json.loads(l) for l in open(root / "synthetic/autolabel.jsonl")]
    assert [o["status"] for o in out] == ["ok", "ok", "rejected", "ok", "rejected"]
    assert "gas_cylinder: detector IoU 0.00" in out[2]["reason"] and "changed by" in out[4]["reason"]
    assert seen == ["fire extinguisher", "fire extinguisher", "gas cylinder"]  # scene_edits never call the detector
    assert (root / "synthetic/labels/0.txt").read_text().startswith("0 0.300000 0.300000 0.400000 0.400000")
    assert (root / "synthetic/labels/1.txt").read_text().startswith("1 ")  # fake_extinguisher keeps its own id
    assert not (root / "synthetic/labels/changed.txt").exists()
    # resume: nothing is relabeled
    assert al.run(CFG, root, NAMES, PARENTS, detect) == {"ok": 0, "rejected": 0}


def test_unknown_class_is_an_error(tmp_path):
    root = make_root(tmp_path, [{"image": "synthetic/0.png", "method": "add_objects", "boxes": [box("tv", [0, 0, 9, 9])]}])
    with pytest.raises(ValueError):
        al.run(CFG, root, NAMES, PARENTS, lambda i, p: [])
