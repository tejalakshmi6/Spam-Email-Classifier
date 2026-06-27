from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# load the saved model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    email = request.form["email"]

    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)
    probability = model.predict_proba(email_vector)
    confidence = max(probability[0]) * 100
    confidence = round(confidence, 2)

    if prediction[0] == 1:
        result = "🚫 Spam"
        result_class = "spam"
    else:
        result = "✅ Not Spam"
        result_class = "ham"

    return render_template(
    "index.html",
    prediction=result,
    confidence=confidence,
    result_class=result_class,
    email=email
)

if __name__ == "__main__":
    app.run(debug=True)