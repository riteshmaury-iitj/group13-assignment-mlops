# AG News Text Classification - DistilBERT

## Model Description

This is a fine-tuned DistilBERT model for 4-class text classification on the AG News dataset. The model classifies news articles into World, Sports, Business, and Sci/Tech categories.

- **Developed by:** Yuvaraj Kosuru (IIT Jodhpur, Group 13)
- **Model type:** Text Classification (Sequence Classification)
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** distilbert-base-uncased

## Model Sources
- **Repository:** https://github.com/riteshmaury-iitj/group13-assignment-mlops
- **HuggingFace Model:** https://huggingface.co/YuvarajK-g25ait2054/ag-news-distilbert

## Uses

### Direct Use
This model can be used directly for classifying English news articles into one of four categories:
- **0:** World
- **1:** Sports  
- **2:** Business
- **3:** Sci/Tech

### Downstream Use
The model can be integrated into:
- News aggregation platforms
- Content recommendation systems
- Automated news categorization pipelines
- Document organization tools

### Out-of-Scope Use
- Non-English text classification
- News categories beyond the 4 trained classes
- Real-time critical decision-making systems
- Content moderation or filtering

## Bias, Risks, and Limitations

- The model is trained on AG News dataset which may contain biases from the source articles
- Performance may degrade on news articles from domains not well-represented in AG News
- Model may struggle with ambiguous articles that could fit multiple categories
- Not suitable for production use without thorough testing and validation

### Recommendations
Users should be aware that this is an academic project model and should validate performance on their specific use case before deployment.

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "YuvarajK-g25ait2054/ag-news-distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example usage
text = "The stock market rallied today as tech companies reported strong earnings."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

# Map to label
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
print(f"Category: {id2label[predicted_class]}")
```

## Training Details

### Training Data
- **Dataset:** AG News Corpus
- **Size:** 120,000 training samples, 7,600 test samples
- **Classes:** 4 (World, Sports, Business, Sci/Tech)
- **Source:** HuggingFace Datasets (`ag_news`)
- **Preprocessing:** 
  - Lowercased text
  - Removed URLs and HTML tags
  - Removed punctuation
  - Normalized whitespace

### Training Procedure

**Two versions were trained and compared:**

**Version 1 (Best Model):**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Weight decay: 0.01
- Max sequence length: 128

**Version 2:**
- Learning rate: 5e-5
- Batch size: 32
- Epochs: 3
- Weight decay: 0.01
- Max sequence length: 128

### Training Hyperparameters
- **Optimizer:** AdamW
- **Training regime:** fp32 (float32)
- **Evaluation strategy:** Per epoch
- **Best model selection:** Based on F1 score (weighted)

### Speeds, Sizes, Times
- **Hardware:** Kaggle GPU T4 x2
- **Framework:** PyTorch 2.5.1, Transformers 4.46.0
- **Training time:** ~30-40 minutes per version
- **Model size:** ~260 MB (DistilBERT base)
- **Training speed:** ~4-5 iterations/second

## Evaluation

### Testing Data
- **Dataset:** AG News test split (7,600 samples)
- **Same preprocessing** as training data

### Metrics
- **Accuracy:** Overall classification accuracy
- **F1 Score:** Weighted F1 score across all 4 classes
- **Loss:** Cross-entropy loss

### Results

**Version 1 (Selected - Best Model):**
- Accuracy: **94.25%** (0.9425)
- F1 Score (weighted): **0.9425**
- Evaluation Loss: 0.4144

**Version 2:**
- Accuracy: **94.12%** (0.9412)
- F1 Score (weighted): **0.9412**
- Evaluation Loss: 0.3542

**Summary:**
Version 1 achieved slightly better accuracy and F1 score, making it the selected model for deployment. Both versions demonstrate strong performance on the AG News classification task with >94% accuracy.

**Note:** Detailed metrics and training curves available at W&B project: `mlops-assignment3`

## Environmental Impact

- **Hardware Type:** NVIDIA Tesla T4 GPU (x2)
- **Cloud Provider:** Kaggle
- **Compute Region:** US (Kaggle default)
- **Training Time:** ~1-1.5 hours total (both versions)

## Technical Specifications

### Model Architecture and Objective
- **Base Model:** DistilBERT (distilbert-base-uncased)
- **Parameters:** ~66 million
- **Objective:** Multi-class text classification (4 classes)
- **Architecture:** Transformer encoder with classification head

### Compute Infrastructure

**Hardware:**
- 2x NVIDIA Tesla T4 GPUs (16GB each)
- Kaggle notebook environment

**Software:**
- Python 3.10
- PyTorch 2.5.1
- Transformers 4.46.0
- Datasets 2.14.0
- Weights & Biases for experiment tracking

## Citation

**BibTeX:**
```bibtex
@misc{ag-news-distilbert-2026,
  author = {Kosuru, Yuvaraj},
  title = {AG News Text Classification with DistilBERT},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/YuvarajK-g25ait2054/ag-news-distilbert}}
}
```

**APA:**
Kosuru, Y. (2026). AG News Text Classification with DistilBERT. HuggingFace Model Hub. https://huggingface.co/YuvarajK-g25ait2054/ag-news-distilbert

## Model Card Authors

Yuvaraj Kosuru (g25ait2054@iitj.ac.in)  
IIT Jodhpur - MLOps Assignment, Group 13

## Model Card Contact

For questions or feedback, please open an issue in the GitHub repository:  
https://github.com/riteshmaury-iitj/group13-assignment-mlops/issues
