import json
import re
import string
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# utility functions for the project
# task 3 stuff


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

    print(f"Number of labels: {model.config.num_labels}")
    print(f"Labels: {model.config.id2label}")
    return model, tokenizer


def create_pipeline(model_name="distilbert-base-uncased"):
    # creates huggingface pipeline - handles device + token_type_ids automatically
    # same approach as inference.ipynb Step 3
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # DistilBERT does not use token_type_ids — remove it from tokenizer output
    if hasattr(tokenizer, 'model_input_names') and 'token_type_ids' in tokenizer.model_input_names:
        tokenizer.model_input_names = [x for x in tokenizer.model_input_names if x != 'token_type_ids']

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None
    )
    print("Inference pipeline ready!")
    return classifier


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
    print("clean_text test:", clean_text("Stock market <b>RISES</b>! http://test.com"))
