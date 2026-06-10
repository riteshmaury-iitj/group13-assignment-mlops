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


# exact 4 texts from inference.ipynb Cell 7
DEMO_TEXTS = [
    "The stock market surged today as investors reacted to positive earnings reports from major tech companies.",
    "NASA successfully launched a new satellite to study climate change patterns in the Arctic region.",
    "Manchester United defeated Liverpool 3-1 in an exciting Premier League match at Old Trafford.",
    "The United Nations Security Council held an emergency meeting to discuss the ongoing conflict in the region."
]


def main():
    parser = argparse.ArgumentParser(description="AG News text classifier")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to classify (omit to run --demo)")
    parser.add_argument("--demo", action="store_true",
                        help="Run all 4 sample texts from inference.ipynb Cell 7")
    parser.add_argument("--model", type=str,
                        default="YuvarajK-g25ait2054/ag-news-distilbert",
                        help="HuggingFace model name")
    parser.add_argument("--labels", type=str, default="id2label.json",
                        help="Path to id2label.json")
    args = parser.parse_args()

    # default to demo if no text provided
    if not args.text and not args.demo:
        args.demo = True

    # load label mapping
    id2label = load_label_mapping(args.labels)
    print(f"Categories: {list(id2label.values())}")
    print(f"Model: {args.model}\n")

    # build pipeline — exact approach from inference.ipynb Cell 7
    classifier = create_pipeline(args.model)

    if args.demo:
        # run all 4 notebook texts (inference.ipynb Cell 7)
        print("=" * 80)
        print("Demo — Custom Text Inference Results (inference.ipynb Cell 7)")
        print("=" * 80)
        texts_cleaned = [clean_text(t) for t in DEMO_TEXTS]
        predictions = classifier(texts_cleaned, truncation=True, max_length=128)
        all_results = []
        for text, cleaned, pred in zip(DEMO_TEXTS, texts_cleaned, predictions):
            top_pred = max(pred, key=lambda x: x["score"])
            all_scores = sorted(pred, key=lambda x: x["score"], reverse=True)
            score_str = ", ".join(f"{p['label']}: {p['score']:.3f}" for p in all_scores)
            print(f"\nOriginal:  {text[:80]}...")
            print(f"Cleaned:   {cleaned[:80]}...")
            print(f"  Predicted : {top_pred['label']} (confidence: {top_pred['score']:.4f})")
            print(f"  All scores: {score_str}")
            all_results.append({
                "text": text, "cleaned_text": cleaned,
                "prediction": top_pred["label"],
                "confidence": round(float(top_pred["score"]), 4),
                "all_scores": [{"category": p["label"], "confidence": round(float(p["score"]), 4)} for p in all_scores]
            })
        print("\n" + "=" * 80)
        output_file = "data/inference_result.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved to: {output_file}")
    else:
        # single text mode
        print(f"Input: {args.text[:100]}...\n")
        result = classify_text(args.text, classifier)
        print("=" * 80)
        print("Custom Text Inference Results:")
        print("=" * 80)
        print(f"\nOriginal : {args.text[:80]}...")
        print(f"Cleaned  : {result['cleaned_text'][:80]}...")
        print(f"  Predicted : {result['prediction']} (confidence: {result['confidence']:.4f})")
        score_str = ", ".join(f"{s['category']}: {s['confidence']:.3f}" for s in result['all_scores'])
        print(f"  All scores: {score_str}")
        print("=" * 80)
        output_file = "data/inference_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

