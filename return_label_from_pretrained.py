from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import sys

def predict_label_from_pretrained(text,fine_tunned_name):
    """ return the label predicted from the text as input."""
    loaded_tokenizer = AutoTokenizer.from_pretrained(fine_tunned_name)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(fine_tunned_name)
    nlp = TextClassificationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)
    return nlp(text, truncation=True)

if __name__ == '__main__':
    text = sys.argv[1]
    fine_tunned_name = "lincoln/flaubert-mlsum-topic-classification"
    print(predict_label_from_pretrained(text,fine_tunned_name))