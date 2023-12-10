import json
import torch
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with open("config.json", "r") as config_file:
    config = json.load(config_file)

def generate_test_data(tokenizer):
    batch_size = config["batch_size"]

    test = pd.read_csv("data/val_split.csv")
    test_sentences = list(test["Messages"])
    test_labels = list(test["gen_label"])
    test_tokens = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    test_dataset = TensorDataset(test_tokens["input_ids"], test_tokens["attention_mask"], torch.tensor(test_labels))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader

def test(model, test_dataloader):
    model.eval()

    # Lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []

    # Iterate through the test dataset
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()

            all_predictions.extend(predictions)
            all_true_labels.extend(labels.tolist())

    # Calculate and print metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    confusion_mat = confusion_matrix(all_true_labels, all_predictions)
    classification_rep = classification_report(all_true_labels, all_predictions)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Classification Report:")
    print(classification_rep)

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_dataloader = generate_test_data(tokenizer)
    test(model, test_dataloader)