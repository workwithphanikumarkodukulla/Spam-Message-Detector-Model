from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer











nltk.data.path.append("/opt/render/nltk_data")
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir="/opt/render/nltk_data")
    stop_words = set(stopwords.words("english"))
app = Flask(__name__)
model = joblib.load("spam_classifier.pkl")
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    message = ""
    if request.method == "POST":
        if request.is_json:
            message = request.get_json().get("message", "")
            cleaned = preprocess_text(message)
            result = model.predict([cleaned])[0]
            prediction = "Spam" if result == 1 else "Ham"
            return jsonify({"prediction": prediction})
        else:
            message = request.form.get("message", "")
            cleaned = preprocess_text(message)
            result = model.predict([cleaned])[0]
            prediction = "Spam" if result == 1 else "Ham"
    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)