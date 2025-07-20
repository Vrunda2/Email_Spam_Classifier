from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open('model/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        email = request.form['email']
        cleaned = clean_text(email)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1] * 100
        prediction = f"{'Spam' if result == 1 else 'Ham'} (Spam Probability: {probability:.2f}%)"
    return render_template('index.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=10000)

