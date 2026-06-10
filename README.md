# Group 13 — MLOps Assignment

## Task
Fine-tune **DistilBERT** on the **AG News** dataset (4-class text classification) with experiment tracking via Weights & Biases and model deployment to Hugging Face Hub.

## Project Links
| Resource | Link |
|---|---|
| GitHub Repo | [riteshmaury-iitj/group13-assignment-mlops](https://github.com/riteshmaury-iitj/group13-assignment-mlops) |
| HuggingFace Model | [YuvarajK-g25ait2054/ag-news-distilbert](https://huggingface.co/YuvarajK-g25ait2054/ag-news-distilbert) |
| Docker Image | [yuvarajkg25ait2054/ag-news-classifier](https://hub.docker.com/r/yuvarajkg25ait2054/ag-news-classifier) |
| W&B Dashboard | [mlops-assignment3](https://wandb.ai/g25ait2054-iit-jodhpur/mlops-assignment3) |
| Kaggle Notebook | [group13-ag-news-k](https://www.kaggle.com/code/kyuvarajg25ait2054/group13-ag-news-k) |

## Results
| Version | Learning Rate | Batch Size | Epochs | Accuracy | F1 Score |
|---|---|---|---|---|---|
| run-v1 (best) | 2e-5 | 16 | 3 | **94.34%** | **0.9434** |
| run-v2 | 5e-5 | 64 | 5 | 94.12% | 0.9412 |

## Repository Structure
```
├── prepare_data.ipynb          # Data inspection, cleaning, label encoding
├── train_model.ipynb           # Training (Kaggle), W&B tracking, HF push
├── inference.py                # CLI inference script
├── utils.py                    # Model loading utilities
├── Dockerfile                  # Docker container for inference
├── requirements.txt            # Python dependencies
├── id2label.json               # Label mapping (0→World, 1→Sports, 2→Business, 3→Sci/Tech)
├── sample_data.json            # Sample test inputs
└── .github/workflows/
    ├── ci.yml                  # CI pipeline (runs on push/PR)
    └── inference.yml           # Inference workflow (runs on push/PR)
```

## Quick Start

### Run inference with Docker
```bash
docker pull yuvarajkg25ait2054/ag-news-classifier:latest
docker run yuvarajkg25ait2054/ag-news-classifier:latest \
  --text "Stock market rises on strong earnings" \
  --model "YuvarajK-g25ait2054/ag-news-distilbert"
```

### Run inference locally
```bash
pip install -r requirements.txt
python inference.py --text "NASA launches new Mars mission" \
  --model "YuvarajK-g25ait2054/ag-news-distilbert"
```

## Kaggle Secrets Required
| Secret | Source |
|---|---|
| `wb_key` | [wandb.ai/authorize](https://wandb.ai/authorize) |
| `hf_key` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `GITHUB_TOKEN` | GitHub PAT with `repo` scope |

## Team
Group 13 — IIT Jodhpur | PGD AI Program
