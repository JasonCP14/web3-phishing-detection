import torch

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the fine-tuned BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set the model in evaluation mode
model.eval()

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in the request"}), 400

    text = data["text"]

    # Tokenize and preprocess the input
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    result = "Positive" if predicted_class == 1 else "Negative"
    proba = round(max(probabilities[0]).item(), 2)

    return jsonify({"result": result, "proba": proba})

if __name__ == "__main__":
    app.run(debug=True)
