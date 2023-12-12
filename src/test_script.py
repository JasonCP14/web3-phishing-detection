from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import json
import torch
import numpy as np
import pandas as pd


with open("./config.json", "r") as config_file:
    config = json.load(config_file)

def preprocess(test_data: pd.DataFrame) -> pd.DataFrame:
    """Removes leading and trailing white spaces and quotation marks.

    Args:
        test_data (pd.DataFrame): Test data

    Returns:
        pd.DataFrame: Preprocessed test data
    """

    test_data["Messages"] = test_data.Messages.str.strip("\"'\s")
    return test_data

def generate_test_data(test_data: pd.DataFrame, tokenizer: BertTokenizer) -> DataLoader:
    """Converts the test data into a test dataloader.

    Args:
        test_data (pd.DataFrame): Test data
        tokenizer (BertTokenizer): Bert Tokenizer

    Returns:
        DataLoader: Test dataloader
    """

    batch_size = config["batch_size"]

    test_sentences = list(test_data["Messages"])
    test_labels = list(test_data["gen_label"])
    test_tokens = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    test_dataset = TensorDataset(test_tokens["input_ids"], test_tokens["attention_mask"], torch.tensor(test_labels))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader

def test(model: BertForSequenceClassification, test_dataloader: DataLoader) -> None:
    """Evaluate the model on the test data.

    Args:
        model (BertForSequenceClassification): BERT model
        test_dataloader (DataLoader): Test dataloader
    """

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
    seed_val = 2024
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Retrieve BERT model and tokenizer
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained("../docker/backend/saved_model")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load the test data
    test_data = pd.read_csv("../data/test_data.csv")
    test_data = preprocess(test_data)
    test_dataloader = generate_test_data(test_data, tokenizer)

    # Test the model
    test(model, test_dataloader)