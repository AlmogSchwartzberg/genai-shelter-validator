# GenAI Shelter Validator

Course project for *Deep Generative Models for Visual Data* (HIT).
We test whether synthetic images generated with Stable Diffusion improve light fine-tuning of existing Hugging Face object detection models, measured on real images only. Domain: shelters. The models detect the objects a valid shelter must contain and lookalikes that appear valid but are not. Full requirements: [PRD.md](PRD.md).

## Structure

| Path | Purpose |
|---|---|
| `config.yaml` | All parameters: domain, models, conditions, amounts, training recipe, seed |
| `Data/objects_table.csv` | Required objects, lookalikes and forbidden objects: appearance, cause, size, how to generate |
| `Code/common.py` | Shared helpers: split, labels, mode construction, metrics, verdict rule (one copy, so notebooks agree) |
| `Code/00_environment_check.ipynb` | Checks GPU, packages, Drive, model names, LLM key, generation and detection |
| `Code/01_generate.ipynb` | LLM attributes, then method A (add objects) and method B (change capture conditions) |
| `Code/02_validate.ipynb` | Realism report: filter, FID, Inception Score, coverage, eye test |
| `Code/03_finetune.ipynb` | Fine-tune every detection model in 5 modes |
| `Code/04_results.ipynb` | Zero-shot baseline, metrics, shelter verdict, plots |

Images, checkpoints and models are stored in Google Drive (`drive_root` in `config.yaml`), never in Git. Drive holds three folders, each created when first needed:

| Drive folder | Content |
|---|---|
| `real/` | `images/` and `annotations.json` (COCO: `images[].source_url`, optional `images[].condition`, category names = names in `objects_table.csv`). Written by the team. `splits.json` is written by the notebooks |
| `synthetic/` | `images/`, one `meta/<id>.json` per image (method, condition, prompt, attributes, seed, model, boxes, filter result), `attributes.json` |
| `results/` | `models/`, `plots/`, `experiments.csv`, `realism_report.json`, ... |

## Getting started

1. Open [`00_environment_check.ipynb` in Colab](https://colab.research.google.com/github/AlmogSchwartzberg/genai-shelter-validator/blob/dev/Code/00_environment_check.ipynb).
2. Runtime > Change runtime type > **T4 GPU**.
3. Add a Colab secret `openai_key` (key icon in the left bar, switch on notebook access).
4. Runtime > Run all. The last cell prints a pass/fail summary.

## Running the pipeline

1. Put the labeled real images in Drive `real/` (see above). Without them method B and notebooks 02-04 cannot run.
2. `01_generate`: the first run writes `synthetic/attributes.json` and stops. Review it, set `"reviewed": true`, copy it to `Data/attributes.json`, run again. A rerun after a disconnect continues where it stopped.
3. `02_validate`: run before training. It marks every synthetic image as passed or rejected; later notebooks use only passed images.
4. `03_finetune`: 3 models x 5 modes with checkpoints in Drive; finished runs are skipped.
5. `04_results`: tables and plots in Drive `results/`. Copy what you need into `Results/` and `Visuals/`.

Modes: 1 real only, 2 real + A, 3 real + B, 4 real + A + B, 5 real + classical augmentation of the same images as mode 3. Modes 2-5 add the same number of extra images (`training.synthetic_per_mode`).

## Workflow

Each team member works on their own branch and opens a PR to `dev`. Keep each notebook in its own file to avoid merge conflicts. To run a notebook from a branch, set `BRANCH` in its first cell.
