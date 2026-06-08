import argparse
import json
import torch
from transformers import pipeline

# inference script for ag news classification
# categories: world, sports, business, sci/tech

def classify_text(text, model_name, id2label):
    # classify news article using the model
    
    # setup pipeline
    device_num = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device_num,
        return_all_scores=True
    )
    
    # get predictions
    results = classifier(text)[0]
    
    # convert LABEL_0, LABEL_1 etc to actual category names
    mapped_results = []
    for r in results:
        label_id = r['label'].replace('LABEL_', '')
        cat_name = id2label.get(label_id, "unknown")
        mapped_results.append({
            "category": cat_name,
            "confidence": r['score']
        })
    
    # sort by confidence
    mapped_results = sorted(mapped_results, key=lambda x: x['confidence'], reverse=True)
    
    output = {
        "text": text,
        "prediction": mapped_results[0]['category'],
        "confidence": mapped_results[0]['confidence'],
        "all_scores": mapped_results
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(description="AG News classifier")
    parser.add_argument("--text", type=str, required=True, help="text to classify")
    parser.add_argument("--model", type=str, default="Recurrent/ag-news-distilbert", 
                       help="model name")
    parser.add_argument("--labels", type=str, default="id2label.json", 
                       help="label file")
    
    args = parser.parse_args()
    
    # load labels
    with open(args.labels, "r") as f:
        id2label = json.load(f)
    
    print(f"Input: {args.text[:100]}...")
    print(f"Model: {args.model}")
    print(f"Categories: {list(id2label.values())}\n")
    
    # run classification
    result = classify_text(args.text, args.model, id2label)
    
    # show results
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
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
