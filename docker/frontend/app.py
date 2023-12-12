from flask import Flask, render_template, request

import requests

app = Flask(__name__)

BACKEND_URL = "http://backend:5001"

@app.route("/")
def index() -> str:
    """Renders the index page
    
    Returns:
        str: The rendered index page
    """

    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify() -> str:
    """Accepts the user input, sends it into the backend for classification, and renders the result page.

    Returns:
        str: The rendered result page
    """

    if request.method == "POST":
        user_input = request.form["user_input"]

        # Make a request to the backend API endpoint
        response = requests.post(f"{BACKEND_URL}/classify", json={"text": user_input})

        if response.status_code == 200:
            result = response.json()["result"]
            proba = response.json()["proba"]
            
        else:
            result = "Unknown"
            proba = "Unknown"

        return render_template("result.html", user_input=user_input, result=result, proba=proba)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
