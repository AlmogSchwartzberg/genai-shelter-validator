# GenAI Shelter Validator

Course project for *Deep Generative Models for Visual Data* (HIT).
We test whether synthetic images generated with Stable Diffusion improve light fine-tuning of existing Hugging Face object detection models, measured on real images only. Domain: shelters. The models detect the objects a valid shelter must contain and lookalikes that appear valid but are not. Full requirements: [PRD.md](PRD.md).

## Structure

| Path | Purpose |
|---|---|
| `config.yaml` | All parameters: domain, models, conditions, amounts, seed |
| `Data/objects_table.csv` | Required objects and their lookalikes: appearance, cause, size, how to generate |
| `Code/00_environment_check.ipynb` | Checks GPU, packages, Drive, generation and detection |
| `Code/01_generate.ipynb` | Generate synthetic images (attribute list from an LLM, method A, method B) |
| `Code/synth.py` | Generation logic used by notebook 01 (test: `pytest Code/test_synth.py`, no GPU needed) |
| `Data/attributes.json` | The approved LLM attribute list, committed after the team reviewed it. Notebook 01 works on `synthetic/attributes.json` in Drive and refuses to generate until a person sets `"reviewed": true` |
| `Code/autolabel.py` | Verifies generated boxes with Grounding-DINO and writes YOLO labels; proposes boxes for real photos (`real/prelabels.json`) for a person to review (called from notebooks 01 and 02; input format in its docstring; test: `pytest Code/test_autolabel.py`) |
| `Code/02_validate.ipynb` | Realism checks |
| `Code/03_finetune.ipynb` | Fine-tune detection models |
| `Code/04_results.ipynb` | Results table and plots |

Images, checkpoints and models are stored in Google Drive (`drive_root` in `config.yaml`), never in Git. Drive holds three folders, each created when first needed: `real`, `synthetic`, `results`.

## Getting started

1. Open [`00_environment_check.ipynb` in Colab](https://colab.research.google.com/github/AlmogSchwartzberg/genai-shelter-validator/blob/dev/Code/00_environment_check.ipynb).
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > Run all. The last cell prints a pass/fail summary.

## API keys

Keys live in a `.env` file, never in Git (`.env` is in `.gitignore`). Copy `.env.example` to `.env` and fill it in. In Colab put the file in the Drive folder (`drive_root`), because the repo is cloned fresh on every run; locally put it in the repo root. Notebook 01 reads `OPENAI_API_KEY` from it; a real environment variable or a Colab secret of the same name also works.

## Workflow

Each team member works on their own branch and opens a PR to `dev`. Keep each notebook in its own file to avoid merge conflicts. To run a notebook from a branch, set `BRANCH` in its first cell.

## Synthetic data format

Written by `Code/01_generate.ipynb` under `drive_root`. Real photos go in `real/` (`*.jpg` and `annotations.json`, COCO, boxes proposed by `autolabel.prelabel` (`real/prelabels.json`) and reviewed by hand, class names as in `Data/objects_table.csv`); notebook 01 creates `real/split.json` once (seed 42, `train` / `val` / `test` lists of file names). Generation starts only from `train`. Photos are fitted into 512x512 without stretching (gray 114 padding, like YOLO letterboxing); boxes are shifted and scaled with the photo, and nothing is painted in the padding. `annotations.json` must give each photo's true width and height.

| Path | Content |
|---|---|
| `synthetic/images/<id>.png` | Generated image, 512x512. Dropped images stay here too |
| `synthetic/metadata.jsonl` | One JSON per image that **passed** the generation filter |
| `synthetic/rejected.jsonl` | Same record for images the filter dropped; the reason is in `filter.reasons` |
| `synthetic/source_512/` | The real photo each image started from, fitted to 512x512 (file name `<stem>_<hash of the path>.png`) |
| `synthetic/attributes.json` | Working copy of the attribute list |

Class id = row order of `Data/objects_table.csv`, starting at 0. Ids look like `fire_extinguisher_0007` (method `add_objects`) or `add_people_0012` (method `scene_edits`).

```json
{"id": "fake_extinguisher_0003", "image": "synthetic/images/fake_extinguisher_0003.png",
 "method": "add_objects", "edit_type": null,
 "source": "synthetic/source_512/p12.png", "source_real": "real/p12.jpg",
 "boxes": [{"class": "fire_extinguisher", "xyxy": [256.0, 256.0, 307.2, 384.0], "origin": "source"},
           {"class": "fake_extinguisher", "xyxy": [40, 300, 96, 440], "origin": "synthetic"}],
 "prompt": "...", "negative_prompt": "...", "attributes": {"variant": "...", "shelter_type": "...", "lighting": "..."},
 "seed": 123, "model": "stable-diffusion-v1-5/stable-diffusion-inpainting", "placement": "inpaint", "strength": 1.0, "factor": null,
 "filter": {"passed": true, "reasons": [], "box_change": 0.0, "paint_change": 0.31}}
```

- `method`: `add_objects` (method A) or `scene_edits` (method B). `edit_type`: `add_people`, `add_furniture`, `room_size`, `room_style` (only for `scene_edits`).
- `boxes` is the complete label set of the image. `origin: "source"` boxes come from the real photo (kept, protected, rescaled for `room_size`); `origin: "synthetic"` is the object that was painted in (only `add_objects`). `xyxy` is in pixels of the saved image.
- `filter` is the generation-time check: pixels inside a protected box changed by more than `box_change_threshold` in the raw model output, a blank image, or a painted area that barely changed. Scene edits also carry the class objects in the negative prompt, but a scene edit can still paint a light, picture or bottle that has no box: look for it in the pilot. The real pixels of protected boxes are copied back after generation, so they are identical to `source`. It does **not** check that the painted object looks right: that is `Code/autolabel.py` plus the manual check in PRD 3.4.
- The job list is deterministic (seed in `config.yaml`): rerunning continues after the last saved image, and raising `images_per_class`, `images_per_edit_type` or `overshoot` only adds images. A record cut off by a disconnect is dropped and redone. Changing the seed, the attributes or the model while records exist stops with an error: use a new `drive_root` or delete `synthetic/`.

