import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# inference script for ag news classification
# categories: world, sports, business, sci/tech

def classify_text(text, model_name, id2label):
    # load stuff
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # gpu check
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)
    mdl.eval()
    
    # tokenize
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128)
    
    # distilbert doesnt need token_type_ids
    if 'token_type_ids' in inp:
        del inp['token_type_ids']
    
    inp = {k: v.to(dev) for k, v in inp.items()}
    
    # predict
    with torch.no_grad():
        out = mdl(**inp)
        probs = torch.nn.functional.softmax(out.logits, dim=-1)[0]
    
    # make results list
    results = []
    for i, p in enumerate(probs):
        cat = id2label.get(str(i), "unknown")
        results.append({"category": cat, "confidence": float(p)})
    
    # sort
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    return {
        "text": text,
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
