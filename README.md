# GenAI Shelter Validator

Course project for *Deep Generative Models for Visual Data* (HIT).
We test whether synthetic images generated with Stable Diffusion improve light fine-tuning of existing Hugging Face object detection models, measured on real images only. Domain: shelters. The models detect the objects a valid shelter must contain and lookalikes that appear valid but are not. Full requirements: [PRD.md](PRD.md).

## Structure

| Path | Purpose |
|---|---|
| `config.yaml` | All parameters: domain, models, conditions, amounts, seed |
| `Data/objects_table.csv` | Required objects and their lookalikes: appearance, cause, size, how to generate |
| `Code/00_environment_check.ipynb` | Checks GPU, packages, Drive, generation and detection |
| `Code/01_generate.ipynb` | Generate synthetic images |
| `Code/02_validate.ipynb` | Realism checks |
| `Code/03_finetune.ipynb` | Fine-tune detection models |
| `Code/04_results.ipynb` | Results table and plots |

Images, checkpoints and models are stored in Google Drive (`drive_root` in `config.yaml`), never in Git. Drive holds three folders, each created when first needed: `real`, `synthetic`, `results`.

## Getting started

1. Open [`00_environment_check.ipynb` in Colab](https://colab.research.google.com/github/AlmogSchwartzberg/genai-shelter-validator/blob/dev/Code/00_environment_check.ipynb).
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > Run all. The last cell prints a pass/fail summary.

## Configuration

Review [config.yaml](config.yaml) before running the notebooks:

| Setting | Purpose |
|---|---|
| `drive_root` | Google Drive directory for images, models and results |
| `seed` | Random seed used by the notebooks |
| `generation.model` | Hugging Face model used for image generation |
| `generation.image_size` | Generated image size |
| `generation.conditions` | Capture conditions to simulate |
| `generation.images_per_class` | Number of synthetic images per class |
| `detection.models` | Hugging Face object detection models to evaluate |

Model choices are still open in the configuration. See [PRD.md](PRD.md) for the experiment requirements.

## Troubleshooting

- **Old `torchao` remains loaded:** In Colab, select **Runtime > Disconnect and delete runtime**, reconnect, select a GPU runtime and run all cells again, as directed by the environment-check notebook.
- **Running code from another branch:** Set `BRANCH` in the repository setup cell before running it. It defaults to `dev`.
- **Using a different Drive location:** Update `drive_root` in `config.yaml`. Authorize the Google Drive mount when Colab prompts you.

## Workflow

Each team member works on their own branch and opens a PR to `dev`. Keep each notebook in its own file to avoid merge conflicts. To run a notebook from a branch, set `BRANCH` in its first cell.
