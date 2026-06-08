import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# inference script for ag news classification
# categories: world, sports, business, sci/tech

def classify_text(text, model_name, id2label):
    # classify news article using the model
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Tokenize input (no token_type_ids for DistilBERT)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Remove token_type_ids if present (DistilBERT doesn't use them)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Get all predictions
    mapped_results = []
    for idx, prob in enumerate(probs):
        cat_name = id2label.get(str(idx), "unknown")
        mapped_results.append({
            "category": cat_name,
            "confidence": float(prob)
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
