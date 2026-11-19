"""Synthetic image generation (PRD 3.4-3.6, 3.11). Plain helpers plus one Stable Diffusion painter;
Code/01_generate.ipynb drives it. Everything except make_painter() runs without a GPU (see test_synth.py).

Drive layout (drive_root):
  real/                       real photos + annotations.json (COCO, boxes drawn by hand); split.json is written here
  synthetic/images/<id>.png   generated image, image_size x image_size
  synthetic/source_512/       resized copy of the real photo each image started from
  synthetic/metadata.jsonl    images that passed the generation filter, one JSON per line (input of Code/autolabel.py)
  synthetic/rejected.jsonl    images the filter dropped, same fields; the png stays on disk (PRD 8)
Repo: Data/attributes.json    LLM attribute list; generation refuses to start until "reviewed" is true (PRD 3.11)

metadata.jsonl line:
  {"id", "image": "synthetic/images/<id>.png", "method": "add_objects"|"scene_edits",
   "edit_type": null|"add_people"|"add_furniture"|"room_size"|"room_style",
   "source": "synthetic/source_512/<stem>.png", "source_real": "real/<file>",
   "boxes": [{"class", "xyxy": [x0,y0,x1,y1] px of the saved image, "origin": "synthetic"|"source"}],
   "prompt", "negative_prompt", "attributes", "seed", "model", "placement", "strength",
   "filter": {"passed", "reasons", "box_change", "paint_change"}}
Class id = row order of Data/objects_table.csv (0-based), the same as Code/autolabel.py.
"""
import csv
import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

EDITS = ("add_people", "add_furniture", "room_size", "room_style")
# Region sizes for people / furniture at 512 px: (width range, height range, how many). Assumed values.
REGIONS = {"add_people": ((40, 100), (120, 260), (1, 2)), "add_furniture": ((100, 240), (60, 150), (1, 1))}
ATTR_LISTS = ("shelter_types", "lighting", "clutter", "people", "furniture", "room_style")


# ---------- objects table, real photos, attributes ----------
def load_objects(path):
    """Rows of objects_table.csv; every column filled (lookalike_of only for lookalikes) and every lookalike
    points to a required object (PRD 3.2, 10)."""
    rows = list(csv.DictReader(open(path, encoding="utf8")))
    by_name = {r["name"]: r for r in rows}
    for r in rows:
        blank = [k for k, v in r.items() if not (v or "").strip() and not (k == "lookalike_of" and r["role"] != "lookalike")]
        if blank:
            raise ValueError(f"{r['name']}: empty column(s) {blank}")
        if r["role"] == "lookalike" and by_name.get(r["lookalike_of"], {}).get("role") != "required":
            raise ValueError(f"{r['name']}: lookalike_of {r['lookalike_of']!r} is not a required object in the table")
    return rows


def size_range(row):
    """'40-160 px (estimate)' -> (40, 160): longer side of the box at 512 px."""
    return tuple(int(v) for v in re.search(r"(\d+)\s*-\s*(\d+)", row["size_range"]).groups())


def load_real(real_dir, names, size):
    """Images listed in real/annotations.json (COCO) -> [{"file", "boxes": [{"class", "xyxy"}]}], boxes scaled
    to size x size (photos are resized without cropping). An image missing from the file is unlabeled: skipped."""
    real_dir = Path(real_dir)
    coco = json.load(open(real_dir / "annotations.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    unknown = sorted(set(cats.values()) - set(names))
    if unknown:
        raise ValueError(f"classes in annotations.json but not in objects_table.csv: {unknown}")
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    items = []
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        if not (real_dir / im["file_name"]).exists():
            raise FileNotFoundError(real_dir / im["file_name"])
        sx, sy = size / im["width"], size / im["height"]
        boxes = []
        for a in anns[im["id"]]:
            x, y, w, h = a["bbox"]
            boxes.append({"class": cats[a["category_id"]], "xyxy": [round(x * sx, 1), round(y * sy, 1), round((x + w) * sx, 1), round((y + h) * sy, 1)]})
        items.append({"file": im["file_name"], "boxes": boxes})
    return items


def load_split(real_dir, items, seed, test=0.2, val=0.1):
    """real/split.json, created once with the seed (PRD 3.3): {"train", "val", "test"} lists of file names.
    Photos added later are in no list until split.json is deleted and rebuilt."""
    path = Path(real_dir) / "split.json"
    if path.exists():
        return json.load(open(path))
    files = sorted(i["file"] for i in items)
    random.Random(seed).shuffle(files)
    n_test = round(len(files) * test)
    n_val = round((len(files) - n_test) * val)
    split = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
    json.dump(split, open(path, "w"), indent=1)
    return split


def validate_attributes(attrs, names):
    """Raises on a wrong structure; returns warnings (negations in phrases: Stable Diffusion ignores 'no'/'without')."""
    for key in ATTR_LISTS:
        if not attrs.get(key) or not all(isinstance(s, str) and s.strip() for s in attrs[key]):
            raise ValueError(f"attributes: {key!r} must be a non-empty list of strings")
    missing = [n for n in names if not attrs.get("object_variants", {}).get(n)]
    if missing:
        raise ValueError(f"attributes: object_variants has no entries for {missing}")
    phrases = [(k, s) for k in ATTR_LISTS for s in attrs[k]] + [(n, s) for n, v in attrs["object_variants"].items() for s in v]
    return [f"{k}: {s!r}" for k, s in phrases if re.search(r"\b(no|not|without|lacking|missing)\b", s, re.I)]


def generate_attributes(cfg, rows, api_key):
    """Ask the LLM (any OpenAI-compatible API, cfg['llm']) for the attribute lists. reviewed stays false."""
    from openai import OpenAI
    llm, n = cfg["llm"], cfg["llm"]["n_per_list"]
    table = "\n".join(f"- {r['name']} ({r['role']}): {r['appearance']}; {r['cause_or_context']}" for r in rows)
    prompt = f"""We build a synthetic image dataset of public shelters (indoor photos) to train object detectors.
Return ONE JSON object with these keys, each a list of {n} short, different English phrases (max 15 words each):
shelter_types (kinds of shelter rooms), lighting (lighting conditions), clutter (how cluttered the room is),
people (people who could stand in the room), furniture (furniture or storage found in a shelter),
room_style (wall and floor materials, colours, ceiling lights),
and object_variants: an object whose keys are exactly the class names below, each a list of {n} phrases that describe
how that object can look in a photo. A lookalike must look right at first glance but show the defect given in its description.
Describe only what is visible. Never use negations ("no", "without", "missing"): image models ignore them.
Classes:
{table}"""
    client = OpenAI(api_key=api_key, base_url=llm.get("base_url"))
    reply = client.chat.completions.create(
        model=llm["model"], messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, temperature=llm.get("temperature", 0.8))
    attrs = {"reviewed": False, "llm_model": llm["model"], **json.loads(reply.choices[0].message.content)}
    return attrs, validate_attributes(attrs, [r["name"] for r in rows])


def load_attributes(path, names):
    """The gate before generation (PRD 10): missing, invalid or not reviewed -> stop."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing: run the attributes cell first")
    attrs = json.load(open(path))
    validate_attributes(attrs, names)
    if attrs.get("reviewed") is not True:
        raise RuntimeError(f'{path} is not reviewed: read it, fix it, then set "reviewed": true')
    return attrs


# ---------- geometry ----------
def overlap(a, b, margin=0):
    return a[0] - margin < b[2] and b[0] - margin < a[2] and a[1] - margin < b[3] and b[1] - margin < a[3]


def place(rng, size, w, h, avoid, y_min=0, margin=8, tries=60):
    """Random w x h box fully inside the image and at least `margin` px away from every box in `avoid`."""
    if w > size or h > size:
        return None
    for _ in range(tries):
        x0, y0 = rng.randint(0, size - w), rng.randint(min(y_min, size - h), size - h)
        box = [x0, y0, x0 + w, y0 + h]
        if not any(overlap(box, a, margin) for a in avoid):
            return box
    return None


def rects_mask(size, boxes):
    mask = np.zeros((size, size), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[int(y0):int(y1), int(x0):int(x1)] = 255
    return Image.fromarray(mask)


def shrink_canvas(img, factor, size):
    """Room-size edit (PRD 3.5): the photo shrinks by 1/factor into the middle of a size x size canvas; the border is
    the area to outpaint. -> canvas, mask (255 = paint), scale, offset (box = box * scale + offset)."""
    inner = round(size / factor)
    off = (size - inner) // 2
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(img.resize((inner, inner), Image.LANCZOS), (off, off))
    mask = np.full((size, size), 255, np.uint8)
    mask[off:off + inner, off:off + inner] = 0
    return canvas, Image.fromarray(mask), inner / size, off


def box_change(a, b, boxes):
    """Largest mean absolute pixel change (0..1) inside any of the boxes between two same-size images."""
    a, b = np.asarray(a.convert("RGB"), np.float32), np.asarray(b.convert("RGB"), np.float32)
    changes = [np.abs(a[int(y0):int(y1), int(x0):int(x1)] - b[int(y0):int(y1), int(x0):int(x1)]).mean() / 255
               for x0, y0, x1, y1 in boxes if y1 > y0 and x1 > x0]
    return float(max(changes, default=0.0))


# ---------- planning (deterministic: same seed, same job list; raising a count only appends jobs) ----------
def make_prompt(kind, attrs, rng, cls=None):
    """-> (prompt, the attribute values it was built from)."""
    shelter, light = rng.choice(attrs["shelter_types"]), rng.choice(attrs["lighting"])
    used = {"shelter_type": shelter, "lighting": light}
    tail = f"{light}, high resolution photo"
    if kind == "add_objects":
        used["variant"] = rng.choice(attrs["object_variants"][cls])
        return f"{used['variant']}, inside a {shelter}, {tail}", used
    if kind == "add_people":
        used["person"] = rng.choice(attrs["people"])
        return f"{used['person']}, standing inside a {shelter}, {tail}", used
    if kind == "add_furniture":
        used["furniture"] = rng.choice(attrs["furniture"])
        return f"{used['furniture']} inside a {shelter}, {tail}", used
    if kind == "room_size":
        used["clutter"] = rng.choice(attrs["clutter"])
        return f"interior of a {shelter}, {used['clutter']}, {tail}", used
    used["room_style"] = rng.choice(attrs["room_style"])
    return f"interior of a {shelter} with {used['room_style']}, {tail}", used


def plan_jobs(cfg, rows, attrs, items, split):
    """Jobs for method add_objects (images_per_class per class) and scene_edits (images_per_edit_type per edit type),
    interleaved so every class and edit type grows together. Counts are multiplied by `overshoot` because the filter
    drops some images. Backgrounds come from the train part of the real split only (PRD 3.5, 10)."""
    g, seed, size = cfg["generation"], cfg["seed"], cfg["generation"]["image_size"]
    train_files = set(split["train"])
    train = [i for i in items if i["file"] in train_files]
    if not train:
        raise ValueError("no labeled real photos in the train split: collect and annotate real photos first")
    groups = []

    def group(gid, method, edit, count, build):
        rng, jobs = random.Random(f"{seed}/{gid}"), []
        for k in range(math.ceil(count * g["overshoot"])):
            for _ in range(20):  # a photo with no free spot for the region is skipped
                src = rng.choice(train)
                job = build(rng, src)
                if job:
                    break
            else:
                raise ValueError(f"{gid}: no free spot found in 20 photos, objects/regions too large for the photos")
            job.update(id=f"{gid}_{k:04d}", method=method, edit_type=edit, source=src["file"], source_boxes=src["boxes"],
                       strength=g.get("edit_strength", {}).get(edit, 1.0))
            job["seed"] = seed + zlib.crc32(job["id"].encode())
            jobs.append(job)
        groups.append(jobs)

    for row in rows:
        name = row["name"]
        if name in g.get("exclude_classes", []):
            continue

        def build_a(rng, src, row=row, name=name):
            lo, hi = size_range(row)
            long_side, aspect = rng.randint(lo, hi), g["aspect"].get(name, 1.0)
            w, h = (round(long_side * aspect), long_side) if aspect < 1 else (long_side, round(long_side / aspect))
            box = place(rng, size, w, h, [b["xyxy"] for b in src["boxes"]])
            if not box:
                return None
            prompt, used = make_prompt("add_objects", attrs, rng, name)
            neg = ", ".join(s for s in (g["negative_prompt"], g.get("class_negative", {}).get(name)) if s)
            return {"cls": name, "paint_boxes": [box], "prompt": prompt, "attributes": used, "negative": neg}

        group(name, "add_objects", None, g["images_per_class"], build_a)

    for edit in g["scene_edits"]:
        def build_b(rng, src, edit=edit):
            job = {"paint_boxes": []}
            if edit in REGIONS:
                (w0, w1), (h0, h1), (n0, n1) = REGIONS[edit]
                avoid = [b["xyxy"] for b in src["boxes"]]
                for _ in range(rng.randint(n0, n1)):
                    box = place(rng, size, rng.randint(w0, w1), rng.randint(h0, h1), avoid, y_min=size // 2)
                    if box:
                        job["paint_boxes"].append(box)
                        avoid.append(box)
                if not job["paint_boxes"]:
                    return None
            if edit == "room_size":
                job["factor"] = rng.choice(g["room_scale_factors"])
            job["prompt"], job["attributes"] = make_prompt(edit, attrs, rng)
            job["negative"] = g["negative_prompt"]
            return job

        group(edit, "scene_edits", edit, g["images_per_edit_type"], build_b)
    return [j for batch in zip_longest(*groups) for j in batch if j]


def long_prompts(jobs, tokenizer, limit=77):
    """Ids of jobs whose prompt is cut by the CLIP token limit (PRD course guidance: 77 tokens)."""
    return [j["id"] for j in jobs if len(tokenizer(j["prompt"]).input_ids) > limit]


# ---------- one image ----------
def render(job, src_img, paint, cfg):
    """Edit one image. paint(canvas, mask, job) -> PIL image of the same size (the Stable Diffusion call).
    -> (final image, metadata record). Pixels outside the mask, protected boxes included, are copied back from the
    canvas, so a box can only 'change' inside the raw model output; that is what the filter measures."""
    g, size = cfg["generation"], cfg["generation"]["image_size"]
    src_boxes = job["source_boxes"]
    if job["edit_type"] == "room_size":
        canvas, mask, scale, off = shrink_canvas(src_img, job["factor"], size)
        keep = [{"class": b["class"], "xyxy": [round(v * scale + off, 1) for v in b["xyxy"]]} for b in src_boxes]
    else:
        canvas, keep = src_img, src_boxes
        if job["edit_type"] == "room_style":  # everything except the protected boxes
            mask = np.full((size, size), 255, np.uint8)
            for x0, y0, x1, y1 in (b["xyxy"] for b in src_boxes):
                mask[int(y0):int(y1), int(x0):int(x1)] = 0
            mask = Image.fromarray(mask)
        else:
            mask = rects_mask(size, job["paint_boxes"])
    raw = paint(canvas, mask, job).convert("RGB")
    protected = [b["xyxy"] for b in keep]
    changed = box_change(canvas, raw, protected)
    inside = np.asarray(mask) > 0
    paint_change = float(np.abs(np.asarray(raw, np.float32) - np.asarray(canvas, np.float32))[inside].mean() / 255)
    reasons = []
    if changed > g["box_change_threshold"]:
        reasons.append(f"pixels inside a protected box changed by {changed:.2f}")
    if np.asarray(raw).std() < 3:
        reasons.append("image is blank (safety checker or failed generation)")
    elif paint_change < g["min_paint_change"]:
        reasons.append(f"painted area barely changed ({paint_change:.3f})")
    new = [{"class": job["cls"], "xyxy": b, "origin": "synthetic"} for b in job["paint_boxes"]] if job["method"] == "add_objects" else []
    boxes = [{**b, "origin": "source"} for b in keep] + new
    stem = Path(job["source"]).with_suffix("").as_posix().replace("/", "_")
    rec = {"id": job["id"], "image": f"synthetic/images/{job['id']}.png", "method": job["method"], "edit_type": job["edit_type"],
           "source": f"synthetic/source_512/{stem}.png", "source_real": f"real/{job['source']}", "boxes": boxes,
           "prompt": job["prompt"], "negative_prompt": job["negative"], "attributes": job["attributes"], "seed": job["seed"],
           "model": g["model"], "placement": g["placement"] if job["method"] == "add_objects" else "inpaint",
           "strength": job["strength"], "factor": job.get("factor"),
           "filter": {"passed": not reasons, "reasons": reasons, "box_change": round(changed, 4), "paint_change": round(paint_change, 4)}}
    return Image.composite(raw, canvas, mask), rec


def run(cfg, root, jobs, paint, limit=None, log=print):
    """Generate every job that has no line yet in metadata.jsonl or rejected.jsonl (resume after a disconnect).
    Image first, record second: a crash in between only repeats that one image. -> number of new images."""
    root, size = Path(root), cfg["generation"]["image_size"]
    syn = root / "synthetic"
    (syn / "source_512").mkdir(parents=True, exist_ok=True)
    (syn / "images").mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for name in ("metadata", "rejected") if (syn / f"{name}.jsonl").exists() for l in open(syn / f"{name}.jsonl")}
    made = 0
    for job in jobs:
        if job["id"] in done:
            continue
        if limit is not None and made >= limit:
            break
        img = Image.open(root / "real" / job["source"]).convert("RGB").resize((size, size), Image.LANCZOS)
        final, rec = render(job, img, paint, cfg)
        copy = root / rec["source"]
        if not copy.exists():
            img.save(copy)
        final.save(root / rec["image"])
        with open(syn / ("metadata.jsonl" if rec["filter"]["passed"] else "rejected.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        made += 1
        if made % 25 == 0:
            log(f"{made} new images, last: {job['id']}")
    return made


def coverage(root, rows):
    """Passed images per generated class and per edit type, and which classes are under images_per_class (PRD 3.7)."""
    syn = Path(root) / "synthetic"
    recs = [json.loads(l) for l in open(syn / "metadata.jsonl")] if (syn / "metadata.jsonl").exists() else []
    per_class = Counter(b["class"] for r in recs for b in r["boxes"] if b["origin"] == "synthetic")
    per_edit = Counter(r["edit_type"] for r in recs if r["method"] == "scene_edits")
    rejected = Counter(re.sub(r"[\d.]+", "#", x) for l in open(syn / "rejected.jsonl") for x in json.loads(l)["filter"]["reasons"]) \
        if (syn / "rejected.jsonl").exists() else Counter()
    return {"per_class": {r["name"]: per_class[r["name"]] for r in rows}, "per_edit_type": dict(per_edit), "rejected_reasons": dict(rejected)}


# ---------- Stable Diffusion (GPU) ----------
def make_painter(cfg):
    """-> paint(canvas, mask, job). add_objects uses generation.placement: inpaint (SD inpainting with the box as mask),
    controlnet (ControlNet inpaint) or gligen (GLIGEN grounded box); scene_edits always use plain SD inpainting
    because they need a free-form mask. Pipelines load on first use. Not run in CI: needs a GPU and Hugging Face."""
    import torch
    from diffusers import AutoPipelineForInpainting, ControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionGLIGENPipeline

    g, size = cfg["generation"], cfg["generation"]["image_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipes = {}

    def pipe(kind):
        if kind not in pipes:
            if kind == "inpaint":
                p = AutoPipelineForInpainting.from_pretrained(g["model"], torch_dtype=dtype, variant="fp16" if device == "cuda" else None)
            elif kind == "controlnet":
                net = ControlNetModel.from_pretrained(g["controlnet"], torch_dtype=dtype)
                p = StableDiffusionControlNetInpaintPipeline.from_pretrained(g["controlnet_base"], controlnet=net, torch_dtype=dtype)
            else:
                p = StableDiffusionGLIGENPipeline.from_pretrained(g["gligen"], torch_dtype=dtype)
            pipes[kind] = p.to(device)
        return pipes[kind]

    def paint(canvas, mask, job):
        kind = g["placement"] if job["method"] == "add_objects" else "inpaint"
        common = dict(prompt=job["prompt"], negative_prompt=job["negative"], num_inference_steps=g["steps"],
                      guidance_scale=g["guidance_scale"], generator=torch.Generator(device).manual_seed(job["seed"]))
        if kind == "inpaint":
            return pipe(kind)(image=canvas, mask_image=mask, strength=job["strength"], height=size, width=size, **common).images[0]
        if kind == "controlnet":
            a, m = np.asarray(canvas, np.float32) / 255, np.asarray(mask.convert("L"), np.float32) / 255
            a[m > 0.5] = -1.0  # masked pixels are -1 in the control image (diffusers ControlNet inpaint recipe)
            control = torch.from_numpy(a[None].transpose(0, 3, 1, 2))
            return pipe(kind)(image=canvas, mask_image=mask, control_image=control, eta=1.0, **common).images[0]
        x0, y0, x1, y1 = job["paint_boxes"][0]
        return pipe(kind)(gligen_phrases=[job["attributes"]["variant"]], gligen_boxes=[[x0 / size, y0 / size, x1 / size, y1 / size]],
                          gligen_inpaint_image=canvas, gligen_scheduled_sampling_beta=1, output_type="pil", **common).images[0]

    return paint
