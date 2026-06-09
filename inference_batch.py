import re
import string
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# batch inference script — loads test data and evaluates on sample
# converted from inference.ipynb
# model: YuvarajK-g25ait2054/ag-news-distilbert
# categories: world, sports, business, sci/tech


def clean_text(text):
    # same cleaning logic as prepare_data.ipynb used during training
    if pd.isna(text) or text is None:
        return ""
    text = text.lower()
    text = re.sub(r'&lt;.*?&gt;', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#?\w+;', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = text.replace('\\', ' ')
    text = re.sub(r'#36;', '$', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Step 1: Load model ────────────────────────────────────────────────────────
HF_REPO_ID = "YuvarajK-g25ait2054/ag-news-distilbert"

print(f"Loading model: {HF_REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)

print(f"Model loaded from: https://huggingface.co/{HF_REPO_ID}")
print(f"Number of labels: {model.config.num_labels}")
print(f"Labels: {model.config.id2label}")

# ── Step 2: Create inference pipeline ────────────────────────────────────────
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None
)
print("Inference pipeline ready!")

# ── Step 3: Load id2label from repo ──────────────────────────────────────────
id2label_url = "https://raw.githubusercontent.com/riteshmaury-iitj/group13-assignment-mlops/main/id2label.json"
import requests
id2label = {int(k): v for k, v in requests.get(id2label_url).json().items()}
print(f"Labels: {id2label}")

# ── Step 4: Load AG News test data ───────────────────────────────────────────
print("\nLoading AG News test data...")
dataset = load_dataset("fancyzhx/ag_news")
test_df = dataset["test"].to_pandas()
test_df.columns = ["text", "label"]

# sample 3 examples per class (12 total)
sample_df = test_df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(n=min(3, len(x)), random_state=42)
).reset_index(drop=True)

print(f"Total test samples: {len(test_df)}")
print(f"Selected sample size: {len(sample_df)}")

# ── Step 5: Run batch inference ───────────────────────────────────────────────
texts = sample_df["text"].tolist()
predictions = classifier(texts, truncation=True, max_length=128)

predicted_labels = []
confidences = []

for pred in predictions:
    top_pred = max(pred, key=lambda x: x["score"])
    predicted_labels.append(top_pred["label"])
    confidences.append(top_pred["score"])

results_df = pd.DataFrame({
    "text": [t[:100] + "..." if len(t) > 100 else t for t in texts],
    "true_label": sample_df["label"].map(id2label).values,
    "predicted_label": predicted_labels,
    "confidence": confidences
})

print("\n" + results_df.to_string())

# ── Step 6: Evaluate accuracy ─────────────────────────────────────────────────
correct = (results_df["true_label"] == results_df["predicted_label"]).sum()
total = len(results_df)
accuracy = correct / total

print(f"\nSample Inference Results:")
print(f"  Correct            : {correct}/{total}")
print(f"  Accuracy           : {accuracy:.2%}")
print(f"  Average Confidence : {results_df['confidence'].mean():.4f}")

# ── Step 7: Custom text inference ─────────────────────────────────────────────
custom_texts = [
    "The stock market surged today as investors reacted to positive earnings reports from major tech companies.",
    "NASA successfully launched a new satellite to study climate change patterns in the Arctic region.",
    "Manchester United defeated Liverpool 3-1 in an exciting Premier League match at Old Trafford.",
    "The United Nations Security Council held an emergency meeting to discuss the ongoing conflict in the region."
]

custom_texts_cleaned = [clean_text(text) for text in custom_texts]

print("\nCustom Text Inference Results:")
print("=" * 80)

custom_predictions = classifier(custom_texts_cleaned, truncation=True, max_length=128)

for text, cleaned, pred in zip(custom_texts, custom_texts_cleaned, custom_predictions):
    top_pred = max(pred, key=lambda x: x["score"])
    print(f"\nOriginal : {text[:80]}...")
    print(f"Cleaned  : {cleaned[:80]}...")
    print(f"Predicted: {top_pred['label']} (confidence: {top_pred['score']:.4f})")
    all_scores = ", ".join(
        f"{p['label']}: {p['score']:.3f}"
        for p in sorted(pred, key=lambda x: -x["score"])
    )
    print(f"All scores: {all_scores}")
