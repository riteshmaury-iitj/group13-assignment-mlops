# Group 13 — MLOps Assignment

## Task
Fine-tune **DistilBERT** on the **AG News** dataset (4-class text classification) with experiment tracking via Weights & Biases and model deployment to Hugging Face Hub.

## Structure
```
├── prepare_data.ipynb     # Data inspection, cleaning, label encoding
├── train_model.ipynb      # Training (Kaggle), W&B tracking, HF push
├── id2label.json          # Label mapping (0→World, 1→Sports, 2→Business, 3→Sci/Tech)
└── .gitignore             # Excludes dataset/ and prepared_data/
```

## Quick Start
1. Run `prepare_data.ipynb` locally to generate cleaned data
2. Upload `prepared_data/` + `id2label.json` as a Kaggle dataset
3. Run `train_model.ipynb` on Kaggle with GPU enabled

## Kaggle Secrets Required
| Secret | Source |
|--------|--------|
| `WANDB_API_KEY` | [wandb.ai/authorize](https://wandb.ai/authorize) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GITHUB_TOKEN` | GitHub PAT with `repo` scope |

## Team
Group 13 — IIT Jodhpur
