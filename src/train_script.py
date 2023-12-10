import json
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

with open("config.json", "r") as config_file:
    config = json.load(config_file)

def generate_train_data(tokenizer):
    batch_size = config["batch_size"]

    train = pd.read_csv("test.csv")[:20]
    train_sentences = list(train["Messages"])
    train_labels = list(train["gen_label"])
    train_tokens = tokenizer(train_sentences, padding=True, truncation=True, return_tensors="pt")

    # Create PyTorch DataLoader
    train_dataset = TensorDataset(train_tokens["input_ids"], train_tokens["attention_mask"], torch.tensor(train_labels))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def train(model, train_dataloader):

    # Set up training parameters
    lr = config["learning_rate"]
    num_epochs = config["num_epochs"]
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Fine-tuning loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    model_name = config["model_name"]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  
    train_dataloader = generate_train_data(tokenizer)
    train(model, train_dataloader)

    # Save the trained model
    model.save_pretrained("saved_model")