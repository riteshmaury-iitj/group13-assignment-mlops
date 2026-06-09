import argparse
import json
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# inference script for ag news classification
# categories: world, sports, business, sci/tech


def clean_text(text):
    # same cleaning logic as prepare_data.ipynb used during training
    if text is None:
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


def classify_text(text, model_name):
    # load model and tokenizer from huggingface hub
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

    print(f"Labels: {mdl.config.id2label}")

    # use pipeline - handles device, token_type_ids etc automatically
    classifier = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None
    )

    # clean text same way as training data
    cleaned = clean_text(text)

    # run prediction
    preds = classifier(cleaned, truncation=True, max_length=128)[0]

    # sort by score
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)

    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": preds[0]['label'],
        "confidence": round(preds[0]['score'], 4),
        "all_scores": [{"category": p['label'], "confidence": round(p['score'], 4)} for p in preds]
    }


def main():
    parser = argparse.ArgumentParser(description="AG News classifier")
    parser.add_argument("--text", type=str, required=True, help="text to classify")
    parser.add_argument("--model", type=str, default="YuvarajK-g25ait2054/ag-news-distilbert",
                        help="huggingface model name")
    parser.add_argument("--labels", type=str, default="id2label.json",
                        help="label file (optional, model config used if not provided)")

    args = parser.parse_args()

    print(f"Input: {args.text[:100]}...")
    print(f"Model: {args.model}\n")

    # run classification
    result = classify_text(args.text, args.model)

    # show results
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
