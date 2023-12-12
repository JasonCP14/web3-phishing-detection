from flask import Flask, jsonify, request, Response
from transformers import BertForSequenceClassification, BertTokenizer

import torch


app = Flask(__name__)

# Load the fine-tuned BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set the model in evaluation mode
model.eval()

@app.route("/classify", methods=["POST"])
def classify() -> Response:
    """Classifies the input sentence using the BERT model.

    Returns:
        Response: The classification and its probability.
    """

    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in the request"}), 400

    # Tokenize and preprocess the input
    text = data["text"].strip("\"'\s")
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
    app.run(debug=True, host="0.0.0.0", port=5001)
