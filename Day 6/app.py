from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]

        if message.strip() == "":
            prediction = "Please enter a message"
        else:
            data = vectorizer.transform([message])

            # Get spam probability
            prob = model.predict_proba(data)[0][1]

            # Adjusted threshold
            if prob >= 0.4:
                prediction = f"Spam Message 🚫 (Confidence: {prob:.2f})"
            else:
                prediction = f"Not Spam ✅ (Confidence: {prob:.2f})"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
