import argparse
import json
import re
import string
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# inference script for ag news classification
# categories: world, sports, business, sci/tech
# usage:
#   single text : python inference.py --text "some news text"
#   batch mode  : python inference.py --text "unused" --batch


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


def load_classifier(model_name):
    # load model and tokenizer from huggingface hub
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(f"Labels: {mdl.config.id2label}")
    # pipeline handles device placement and token_type_ids automatically
    classifier = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None
    )
    print("Inference pipeline ready!\n")
    return classifier


def classify_text(text, classifier):
    # clean text same way as training data
    cleaned = clean_text(text)
    preds = classifier(cleaned, truncation=True, max_length=128)[0]
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": preds[0]['label'],
        "confidence": round(preds[0]['score'], 4),
        "all_scores": [{"category": p['label'], "confidence": round(p['score'], 4)} for p in preds]
    }


def run_batch(classifier):
    # Step 1: load AG News test data (same as notebook Step 4)
    print("Loading AG News test dataset...")
    dataset = load_dataset("fancyzhx/ag_news")
    test_df = dataset["test"].to_pandas()
    test_df.columns = ["text", "label"]

    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    # sample 3 examples per class (12 total)
    sample_df = test_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(3, len(x)), random_state=42)
    ).reset_index(drop=True)

    print(f"Total test samples : {len(test_df)}")
    print(f"Selected sample    : {len(sample_df)}\n")

    # Step 2: run batch inference (notebook Step 5)
    texts = sample_df["text"].tolist()
    predictions = classifier(texts, truncation=True, max_length=128)

    predicted_labels, confidences = [], []
    for pred in predictions:
        top = max(pred, key=lambda x: x["score"])
        predicted_labels.append(top["label"])
        confidences.append(round(top["score"], 4))

    results_df = pd.DataFrame({
        "text": [t[:80] + "..." if len(t) > 80 else t for t in texts],
        "true_label": sample_df["label"].map(id2label).values,
        "predicted_label": predicted_labels,
        "confidence": confidences
    })
    print(results_df.to_string())

    # Step 3: accuracy (notebook Step 6)
    correct = (results_df["true_label"] == results_df["predicted_label"]).sum()
    total = len(results_df)
    print(f"\nSample Accuracy    : {correct}/{total} = {correct/total:.2%}")
    print(f"Avg Confidence     : {results_df['confidence'].mean():.4f}")

    # Step 4: custom texts (notebook Step 7)
    custom_texts = [
        "The stock market surged today as investors reacted to positive earnings reports from major tech companies.",
        "NASA successfully launched a new satellite to study climate change patterns in the Arctic region.",
        "Manchester United defeated Liverpool 3-1 in an exciting Premier League match at Old Trafford.",
        "The United Nations Security Council held an emergency meeting to discuss the ongoing conflict in the region."
    ]

    print("\nCustom Text Inference Results:")
    print("=" * 80)
    custom_preds = classifier([clean_text(t) for t in custom_texts], truncation=True, max_length=128)
    for text, pred in zip(custom_texts, custom_preds):
        top = max(pred, key=lambda x: x["score"])
        all_s = ", ".join(f"{p['label']}: {p['score']:.3f}" for p in sorted(pred, key=lambda x: -x["score"]))
        print(f"\nText      : {text[:80]}...")
        print(f"Predicted : {top['label']} ({top['score']:.4f})")
        print(f"All scores: {all_s}")


def main():
    parser = argparse.ArgumentParser(description="AG News classifier")
    parser.add_argument("--text", type=str, required=True, help="text to classify")
    parser.add_argument("--model", type=str, default="YuvarajK-g25ait2054/ag-news-distilbert",
                        help="huggingface model name")
    parser.add_argument("--batch", action="store_true",
                        help="run batch evaluation on AG News test set")

    args = parser.parse_args()

    # load model once
    classifier = load_classifier(args.model)

    if args.batch:
        # full batch evaluation from notebook
        run_batch(classifier)
    else:
        # single text classification (used by Docker + GitHub Actions)
        print(f"Input : {args.text[:100]}...")
        result = classify_text(args.text, classifier)

        print("=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print("\nAll scores:")
        for item in result['all_scores']:
            print(f"  {item['category']:<12} : {item['confidence']:.4f}")
        print("=" * 50)

        # save to file
        output_file = "inference_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
