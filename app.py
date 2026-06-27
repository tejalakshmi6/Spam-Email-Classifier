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

    if prediction[0] == 1:
        result = "Spam 🚫"
    else:
        result = "Not Spam ✅"

    return result

if __name__ == "__main__":
    app.run(debug=True)