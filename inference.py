import argparse
import json
from utils import clean_text, create_pipeline, load_label_mapping

# inference script for ag news classification
# follows exact approach from inference.ipynb (Cell 15)
# categories: World, Sports, Business, Sci/Tech


def classify_text(text, classifier):
    # Step 1: clean text — same logic as prepare_data.ipynb (notebook Cell 7)
    cleaned = clean_text(text)

    # Step 2: run classifier — notebook Cell 7 approach
    # returns [[{label, score}, ...]] — same format as pipeline(top_k=None)
    pred = classifier(cleaned, truncation=True, max_length=128)[0]

    # Step 3: get top prediction — max score (notebook Cell 7)
    top_pred = max(pred, key=lambda x: x["score"])

    # Step 4: all scores sorted descending
    all_scores = sorted(pred, key=lambda x: x["score"], reverse=True)

    return {
        "text": text,
        "cleaned_text": cleaned,
        "prediction": top_pred["label"],
        "confidence": round(float(top_pred["score"]), 4),
        "all_scores": [
            {"category": p["label"], "confidence": round(float(p["score"]), 4)}
            for p in all_scores
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="AG News text classifier")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to classify")
    parser.add_argument("--model", type=str,
                        default="YuvarajK-g25ait2054/ag-news-distilbert",
                        help="HuggingFace model name")
    parser.add_argument("--labels", type=str, default="id2label.json",
                        help="Path to id2label.json")
    args = parser.parse_args()

    # load label mapping
    id2label = load_label_mapping(args.labels)
    print(f"Categories: {list(id2label.values())}")
    print(f"Model: {args.model}")
    print(f"Input: {args.text[:100]}...\n")

    # build pipeline (notebook Cell 5 + 7)
    classifier = create_pipeline(args.model)

    # run inference (notebook Cell 15)
    result = classify_text(args.text, classifier)

    # print results exactly as notebook Cell 15
    print("=" * 80)
    print("Custom Text Inference Results:")
    print("=" * 80)
    print(f"\nOriginal : {args.text[:80]}...")
    print(f"Cleaned  : {result['cleaned_text'][:80]}...")
    print(f"  Predicted : {result['prediction']} (confidence: {result['confidence']:.4f})")
    score_str = ', '.join(f"{s['category']}: {s['confidence']:.3f}" for s in result['all_scores'])
    print(f"  All scores: {score_str}")
    print("=" * 80)

    # save result to JSON
    output_file = "inference_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

