import argparse
import json
from utils import clean_text, create_pipeline, load_label_mapping

# inference script for ag news classification
# categories: world, sports, business, sci/tech
# uses pipeline approach from inference.ipynb (handles device + token_type_ids)


def classify_text(text, classifier, id2label):
    # clean text same way as training data (notebook Step 7 clean_text logic)
    cleaned = clean_text(text)

    # run prediction using pipeline (notebook Step 3 approach)
    preds = classifier(cleaned, truncation=True, max_length=128)[0]

    # sort by score descending
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)

    # map LABEL_0/LABEL_1 to actual category names via id2label
    results = []
    for p in preds:
        label_idx = p['label'].replace('LABEL_', '')
        cat = id2label.get(label_idx, p['label'])
        results.append({"category": cat, "confidence": round(float(p['score']), 4)})

    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": results[0]['category'],
        "confidence": results[0]['confidence'],
        "all_scores": results
    }


def main():
    parser = argparse.ArgumentParser(description="AG News classifier")
    parser.add_argument("--text", type=str, required=True, help="text to classify")
    parser.add_argument("--model", type=str, default="YuvarajK-g25ait2054/ag-news-distilbert",
                        help="model name")
    parser.add_argument("--labels", type=str, default="id2label.json",
                        help="label file")

    args = parser.parse_args()

    id2label = load_label_mapping(args.labels)

    print(f"Input: {args.text[:100]}...")
    print(f"Model: {args.model}")
    print(f"Categories: {list(id2label.values())}\n")

    classifier = create_pipeline(args.model)
    result = classify_text(args.text, classifier, id2label)

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nAll scores:")
    for item in result['all_scores']:
        print(f"  {item['category']:<12} : {item['confidence']:.4f}")
    print("=" * 50)

    output_file = "inference_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
