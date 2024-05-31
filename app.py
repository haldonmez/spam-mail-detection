from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import joblib
import nltk 
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

def predict_spam(body):
    # Load the model from the file
    lrc = joblib.load('model_files\\spam_model.pkl')
    vectorizer = joblib.load('model_files\\vectorizer.pkl')

    nltk.download("stopwords")

    stemmer = PorterStemmer()
    corpus = []
    stopwords_set = set(stopwords.words("english"))
    stopwords_set.add("subject")
    stopwords_set.add("hou")
    stopwords_set.add("ect")
    stopwords_set.add("enron")

    text = body
    text = text.split()
    text = [word for word in text if word.isalpha() and len(word) > 1]
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = " ".join(text)
    corpus.append(text)

    final_pre = vectorizer.transform(corpus)
    y_pred = lrc.predict(final_pre)

    if y_pred.item() == 1:
        return True
    elif y_pred.item() == 0:
        return False

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body = data.get('body')
    is_spam = predict_spam(body)
    return jsonify({'is_spam': is_spam})

if __name__ == "__main__":
    app.run(port=3000, debug=True)

