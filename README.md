# Group 13 — MLOps Assignment

## Task
Fine-tune **DistilBERT** on the **AG News** dataset (4-class text classification) with experiment tracking via Weights & Biases and model deployment to Hugging Face Hub.


## Results
| Learning Rate | Batch Size | Epochs | Accuracy | F1 Score |
|---|---|---|---|---|
| 2e-5 | 16 | 3 | **94.34%** | **0.9434** |

## Repository Structure
```
├── src/
│   ├── inference.py            # CLI inference script
│   └── utils.py                # Model loading utilities
├── data/
│   ├── id2label.json           # Label mapping (0→World, 1→Sports, 2→Business, 3→Sci/Tech)
│   ├── sample_data.json        # Sample test inputs
│   └── inference_result.json   # Output from inference runs
├── notebooks/
│   ├── prepare_data.ipynb      # Data inspection, cleaning, label encoding
│   ├── train_model.ipynb       # Training (Kaggle), W&B tracking, HF push
│   ├── model_inference.ipynb   # Model inference demonstration
│   └── __notebook__.ipynb      # Notebook template
├── Dockerfile                  # Docker container for inference
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # License
└── .github/workflows/
    ├── ci.yml                  # CI pipeline (runs on push/PR)
    └── inference.yml           # Inference workflow (runs on push/PR)
```

## Quick Start

### Run inference with Docker
```bash
docker pull <your-docker-username>/ag-news-classifier:latest
docker run <your-docker-username>/ag-news-classifier:latest \
  --text "Stock market rises on strong earnings" \
  --model "<your-hf-username>/ag-news-distilbert"
```

### Run inference locally
```bash
pip install -r requirements.txt
python src/inference.py --text "NASA launches new Mars mission" \
  --model "<your-hf-username>/ag-news-distilbert"
```

### Run demo (all 4 sample texts)
```bash
python src/inference.py --demo \
  --model "<your-hf-username>/ag-news-distilbert"
```

## Project Structure

This project follows a clean architecture with organized directories:

- **`src/`** — Production code for inference
  - `inference.py` — Main CLI script for text classification
  - `utils.py` — Utility functions (text cleaning, model loading, label mapping)

- **`data/`** — Data files and outputs
  - `id2label.json` — Label mapping for the 4 AG News categories
  - `sample_data.json` — Sample input texts for testing
  - `inference_result.json` — Generated output from inference runs

- **`notebooks/`** — Jupyter notebooks for development and training
  - Training, data preparation, and inference demonstrations

- **`Dockerfile`** — Container configuration for deployment
  - Pre-downloads model to eliminate internet dependency at runtime

## Kaggle Secrets Required
| Secret | Source |
|---|---|
| `wb_key` | [wandb.ai/authorize](https://wandb.ai/authorize) |
| `hf_key` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GITHUB_TOKEN` | GitHub PAT with `repo` scope |

## Team
Group 13 — IIT Jodhpur | PGD AI Program
