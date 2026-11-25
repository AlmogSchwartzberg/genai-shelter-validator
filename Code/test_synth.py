"""Run: pytest Code/test_synth.py  (no GPU, no model: the painter is a stub)."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import synth

REPO = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(REPO / "config.yaml"))
ROWS = synth.load_objects(REPO / "Data/objects_table.csv")
NAMES = [r["name"] for r in ROWS]
ATTRS = {"reviewed": True, **{k: [f"{k} one", f"{k} two"] for k in synth.ATTR_LISTS},
         "object_variants": {n: [f"{n} variant a", f"{n} variant b"] for n in NAMES}}


def make_real(tmp_path, n=10):
    """n labeled 800x600 photos, one fire extinguisher each, in tmp_path/real."""
    real = tmp_path / "real"
    real.mkdir()
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)).save(real / f"p{i}.jpg")
        images.append({"id": i, "file_name": f"p{i}.jpg", "width": 800, "height": 600})
        anns.append({"id": i, "image_id": i, "category_id": 7, "bbox": [400, 300, 80, 150]})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 7, "name": "fire_extinguisher"}]}, open(real / "annotations.json", "w"))
    return real


def stub(leak=False):
    """Paints the mask white. leak=True also repaints the extinguisher box (256,256,307,352), like a model ignoring the mask."""
    def paint(canvas, mask, job):
        out = np.asarray(canvas).copy()
        out[np.asarray(mask) > 0] = 255
        if leak:
            out[250:360, 250:330] = 0
        return Image.fromarray(out)
    return paint


def small_cfg(**gen):
    cfg = json.loads(json.dumps(CFG))
    cfg["generation"].update({"images_per_class": 2, "images_per_edit_type": 2, "overshoot": 1.0, **gen})
    return cfg


def setup(tmp_path):
    real = make_real(tmp_path)
    items = synth.load_real(real, NAMES, 512)
    split = synth.load_split(real, items, 42)
    return real, items, split


def test_table_is_valid_and_sizes_parse():
    assert len(ROWS) == 10 and synth.size_range(ROWS[0]) == (40, 160)


def test_split_is_deterministic_and_disjoint(tmp_path):
    real, items, split = setup(tmp_path)
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (7, 1, 2)
    assert not set(split["train"]) & set(split["test"])
    (real / "split.json").unlink()
    assert synth.load_split(real, items, 42) == split


def test_real_boxes_are_scaled_to_512(tmp_path):
    _, items, _ = setup(tmp_path)
    assert items[0]["boxes"] == [{"class": "fire_extinguisher", "xyxy": [256.0, 256.0, 307.2, 352.0]}]


def test_plan_is_deterministic_extendable_and_keeps_free_of_existing_boxes(tmp_path):
    _, items, split = setup(tmp_path)
    cfg = small_cfg()
    a = synth.plan_jobs(cfg, ROWS, ATTRS, items, split)
    assert a == synth.plan_jobs(cfg, ROWS, ATTRS, items, split)
    assert len(a) == 10 * 2 + 4 * 2 and {j["source"] for j in a} <= set(split["train"])
    more = synth.plan_jobs(small_cfg(images_per_class=3), ROWS, ATTRS, items, split)
    assert {j["id"]: j for j in a}.items() <= {j["id"]: j for j in more}.items()  # raising a count only appends
    for j in a:
        assert not any(synth.overlap(p, b["xyxy"]) for p in j["paint_boxes"] for b in j["source_boxes"])
    fake = next(j for j in a if j.get("cls") == "fake_extinguisher")
    assert "hose" in fake["negative"]


def test_unreviewed_attributes_stop_generation(tmp_path):
    path = tmp_path / "attributes.json"
    with pytest.raises(FileNotFoundError):
        synth.load_attributes(path, NAMES)
    json.dump({**ATTRS, "reviewed": False}, open(path, "w"))
    with pytest.raises(RuntimeError):
        synth.load_attributes(path, NAMES)
    json.dump(ATTRS, open(path, "w"))
    assert synth.load_attributes(path, NAMES)["reviewed"] is True
    assert synth.validate_attributes({**ATTRS, "lighting": ["dim, no windows"]}, NAMES) == ["lighting: 'dim, no windows'"]


def test_run_resume_contract_and_protected_boxes(tmp_path):
    real, items, split = setup(tmp_path)
    cfg = small_cfg()
    jobs = synth.plan_jobs(cfg, ROWS, ATTRS, items, split)
    assert synth.run(cfg, tmp_path, jobs, stub(), limit=5) == 5  # "disconnect"
    assert synth.run(cfg, tmp_path, jobs, stub()) == len(jobs) - 5  # resume: only the rest
    assert synth.run(cfg, tmp_path, jobs, stub()) == 0
    syn = tmp_path / "synthetic"
    recs = [json.loads(l) for l in open(syn / "metadata.jsonl")]
    assert len(recs) == len(jobs)
    for r in recs:  # the fields Code/autolabel.py reads
        assert (tmp_path / r["image"]).exists() and (tmp_path / r["source"]).exists()
        assert r["method"] in ("add_objects", "scene_edits") and all(b["class"] in NAMES and len(b["xyxy"]) == 4 for b in r["boxes"])
        if r["edit_type"] != "room_size":  # protected boxes are byte-identical to the source copy
            kept = [b["xyxy"] for b in r["boxes"] if b["origin"] == "source"]
            assert synth.box_change(Image.open(tmp_path / r["image"]), Image.open(tmp_path / r["source"]), kept) == 0
    added = next(r for r in recs if r["method"] == "add_objects")
    assert [b["origin"] for b in added["boxes"]] == ["source", "synthetic"]
    cov = synth.coverage(tmp_path, ROWS)
    assert cov["per_class"]["gas_cylinder"] == 2 and cov["per_edit_type"] == {e: 2 for e in synth.EDITS}


def test_room_size_remaps_boxes(tmp_path):
    real, items, split = setup(tmp_path)
    cfg = small_cfg(scene_edits=["room_size"], room_scale_factors=[2.0])
    jobs = [j for j in synth.plan_jobs(cfg, ROWS, ATTRS, items, split) if j["edit_type"] == "room_size"]
    synth.run(cfg, tmp_path, jobs, stub())
    r = json.loads(open(tmp_path / "synthetic/metadata.jsonl").readline())
    assert r["boxes"][0]["xyxy"] == [256.0, 256.0, 281.6, 304.0]  # box * 0.5 + 128
    # the shrunken original survives in the middle of the saved image
    assert synth.box_change(Image.open(tmp_path / r["image"]), synth.shrink_canvas(Image.open(tmp_path / r["source"]), 2.0, 512)[0], [[256, 256, 281, 304]]) == 0


def test_a_leaking_model_or_a_blank_image_is_rejected_with_a_reason(tmp_path):
    real, items, split = setup(tmp_path)
    cfg = small_cfg()
    jobs = [j for j in synth.plan_jobs(cfg, ROWS, ATTRS, items, split) if j["edit_type"] == "add_people"]
    synth.run(cfg, tmp_path, jobs[:1], stub(leak=True))
    blank = lambda canvas, mask, job: Image.new("RGB", canvas.size)
    synth.run(cfg, tmp_path, jobs[1:], blank)
    assert not (tmp_path / "synthetic/metadata.jsonl").exists()
    reasons = [" | ".join(json.loads(l)["filter"]["reasons"]) for l in open(tmp_path / "synthetic/rejected.jsonl")]
    assert "protected box changed by 0.5" in reasons[0] and "blank" in reasons[1]
    assert all((tmp_path / f"synthetic/images/{j['id']}.png").exists() for j in jobs)  # dropped images stay on disk
    assert synth.coverage(tmp_path, ROWS)["rejected_reasons"] == {
        "pixels inside a protected box changed by #": 2, "image is blank (safety checker or failed generation)": 1}


def test_photos_are_fitted_not_stretched(tmp_path):
    real, items, _ = setup(tmp_path)  # 800x600 -> 512x384 inside 512x512, 64 px gray bars above and below
    assert items[0]["pad"] == [[0, 0, 512, 64], [0, 448, 512, 512]]
    assert synth.load_photo(real / "p0.jpg", 512).getpixel((10, 10)) == synth.PAD_GRAY
    sq = {"images": [{"id": 1, "file_name": "p0.jpg", "width": 800, "height": 600}], "categories": [{"id": 7, "name": "gas_cylinder"}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [100, 100, 200, 200]}]}  # a square box
    json.dump(sq, open(real / "annotations.json", "w"))
    x0, y0, x1, y1 = synth.load_real(real, NAMES, 512)[0]["boxes"][0]["xyxy"]
    assert x1 - x0 == pytest.approx(y1 - y0, abs=0.2)
    for bad in ({"width": 0}, {"width": 640}):  # zero size, or not the size of the file
        sq["images"][0].update(bad)
        json.dump(sq, open(real / "annotations.json", "w"))
        with pytest.raises(ValueError):
            synth.load_real(real, NAMES, 512)


def test_nothing_is_painted_in_the_gray_border(tmp_path):
    _, items, split = setup(tmp_path)
    for j in synth.plan_jobs(small_cfg(), ROWS, ATTRS, items, split):
        assert not any(synth.overlap(p, pad) for p in j["paint_boxes"] for pad in j["pad"])


def test_room_style_mask_keeps_a_grid_margin_around_boxes_and_the_border(tmp_path):
    _, items, split = setup(tmp_path)
    cfg = small_cfg()
    job = next(j for j in synth.plan_jobs(cfg, ROWS, ATTRS, items, split) if j["edit_type"] == "room_style")
    seen = {}
    def spy(canvas, mask, job):
        seen["mask"] = np.asarray(mask)
        return canvas
    synth.render(job, Image.new("RGB", (512, 512)), spy, cfg)
    m = seen["mask"]
    assert m[248:360, 248:316].max() == 0  # box (256,256,307,352) plus 8 px, on the 8 px grid
    assert m[:64].max() == 0 and m[448:].max() == 0 and m[200:210, 100:110].min() == 255
    assert synth.grid_round([256, 256, 307.2, 352], 512) == [248, 248, 320, 360]


def test_scene_edits_get_the_class_negative_prompt_and_attributes_are_checked(tmp_path):
    _, items, split = setup(tmp_path)
    jobs = synth.plan_jobs(small_cfg(), ROWS, ATTRS, items, split)
    assert all("ceiling light" in j["negative"] for j in jobs if j["method"] == "scene_edits")
    warn = synth.validate_attributes({**ATTRS, "room_style": ["green walls, bright ceiling lights"], "furniture": ["wooden bench"]}, NAMES)
    assert len(warn) == 1 and "unlabeled class" in warn[0]


def test_split_is_validated(tmp_path):
    real, items, split = setup(tmp_path)
    json.dump({**split, "test": split["test"] + split["train"][:1]}, open(real / "split.json", "w"))
    with pytest.raises(ValueError, match="more than one part"):
        synth.load_split(real, items, 42)
    json.dump({**split, "test": ["nope.jpg"]}, open(real / "split.json", "w"))
    with pytest.raises(ValueError, match="nope.jpg"):
        synth.load_split(real, items, 42)


def test_llm_cannot_approve_its_own_list(tmp_path, monkeypatch):
    import sys, types
    reply = json.dumps({**{k: ["x"] for k in synth.ATTR_LISTS}, "object_variants": {n: ["v"] for n in NAMES}, "reviewed": True})
    fake = types.SimpleNamespace(OpenAI=lambda **kw: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=reply))])))))
    monkeypatch.setitem(sys.modules, "openai", fake)
    attrs, _ = synth.generate_attributes(CFG, ROWS, "key")
    assert attrs["reviewed"] is False


def test_resume_survives_a_cut_line_and_refuses_changed_runs(tmp_path):
    real, items, split = setup(tmp_path)
    cfg = small_cfg()
    jobs = synth.plan_jobs(cfg, ROWS, ATTRS, items, split)
    synth.run(cfg, tmp_path, jobs, stub(), limit=4)
    log = tmp_path / "synthetic/metadata.jsonl"
    whole = log.read_text()
    log.write_text(whole + '{"id": "cut off mid-wr')  # the runtime died while writing
    assert synth.run(cfg, tmp_path, jobs, stub()) == len(jobs) - 4  # the cut record is dropped, its job is not lost or doubled
    assert all(l.startswith("{") and l.endswith("}") for l in log.read_text().splitlines())
    lines = log.read_text().splitlines()
    log.write_text("\n".join(["not json"] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        synth.read_jsonl(log)
    log.write_text("\n".join(lines) + "\n")
    other = json.loads(json.dumps(cfg)); other["seed"] = 7
    with pytest.raises(RuntimeError, match="another seed"):
        synth.run(other, tmp_path, synth.plan_jobs(other, ROWS, ATTRS, items, split), stub())


def test_source_copies_of_a_b_and_a_underscore_b_do_not_collide(tmp_path):
    cfg = small_cfg()
    job = lambda src: {"id": "x", "method": "scene_edits", "edit_type": "add_people", "source": src, "source_boxes": [], "pad": [],
                       "paint_boxes": [[300, 300, 340, 400]], "prompt": "p", "negative": "n", "attributes": {}, "seed": 1, "strength": 1.0}
    rec = lambda src: synth.render(job(src), Image.new("RGB", (512, 512), "gray"), stub(), cfg)[1]["source"]
    assert rec("a/b.jpg") != rec("a_b.jpg")
