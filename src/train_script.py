from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

import json
import pandas as pd
import torch


with open("./config.json", "r") as config_file:
    config = json.load(config_file)

def preprocess(train_data: pd.DataFrame) -> pd.DataFrame:
    """Removes leading and trailing white spaces and quotation marks.

    Args:
        train_data (pd.DataFrame): Training data

    Returns:
        pd.DataFrame: Preprocessed training data
    """

    train_data["Messages"] = train_data.Messages.str.strip("\"'\s")
    return train_data

def generate_train_data(train_data: pd.DataFrame, tokenizer: BertTokenizer) -> DataLoader:
    """Converts the training data into a training dataloader.

    Args:
        train_data (pd.DataFrame): Training data
        tokenizer (BertTokenizer): Bert Tokenizer

    Returns:
        DataLoader: Training dataloader
    """

    batch_size = config["batch_size"]

    # Create tokens and labels from the training data
    train_sentences = list(train_data["Messages"])
    train_labels = list(train_data["gen_label"])
    train_tokens = tokenizer(train_sentences, padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    train_dataset = TensorDataset(train_tokens["input_ids"], train_tokens["attention_mask"], torch.tensor(train_labels))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def train(model: BertForSequenceClassification, train_dataloader: DataLoader) -> None:
    """Trains the model on the training data.

    Args:
        model (BertForSequenceClassification): BERT model
        train_dataloader (DataLoader): Training dataloader
    """

    # Set up training parameters
    lr = config["learning_rate"]
    num_epochs = config["num_epochs"]
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")


if __name__ == "__main__":
    # Initialize BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  

    # Load the training data
    train_data = pd.read_csv("../data/train_split.csv")
    train_data = preprocess(train_data)
    train_dataloader = generate_train_data(train_data, tokenizer)

    # Train the model
    train(model, train_dataloader)

    # Save the trained model
    model.save_pretrained("../docker/backend/saved_model")