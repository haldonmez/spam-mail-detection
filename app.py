from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

# Assuming you have a function `predict_spam` that takes body and returns a boolean
def predict_spam(body):
    # Placeholder for your spam detection logic
    return True

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body = data.get('body')
    is_spam = predict_spam(body)
    return jsonify({'is_spam': is_spam})

if __name__ == "__main__":
    app.run(port=3000, debug=True)

