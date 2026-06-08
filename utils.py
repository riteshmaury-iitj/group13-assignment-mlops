import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# utility functions for the project
# task 3 stuff

def load_model_and_tokenizer(model_name="distilbert-base-uncased"):
    # loads model and tokenizer from huggingface
    # AG News needs 4 classes (world, sports, business, scitech)
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # need to set num_labels=4 for our dataset
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4
    )
    
    print("loaded successfully")
    return model, tokenizer


def load_label_mapping(label_file="id2label.json"):
    # load the json file with label mappings
    f = open(label_file, "r")
    id2label = json.load(f)
    f.close()
    
    return id2label


if __name__ == "__main__":
    # testing
    labels = load_label_mapping()
    print(labels)
    print("total classes:", len(labels))
