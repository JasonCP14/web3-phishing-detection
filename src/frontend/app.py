import json
import torch

from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer

app = Flask(__name__)

with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Load the fine-tuned BERT model and tokenizer
model_name = config["model_name"]
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set the model in evaluation mode
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        user_input = request.form["user_input"]

        # Tokenize and preprocess the input
        tokens = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Convert the result to a human-readable format
        proba = round(max(probabilities[0]).item(), 2)
        result = "Positive" if predicted_class == 1 else "Negative"

        return render_template("result.html", user_input=user_input, proba=proba, result=result)

if __name__ == "__main__":
    app.run(debug=True)