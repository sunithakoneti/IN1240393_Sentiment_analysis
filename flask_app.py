from flask import Flask, render_template, request
import pickle
import sklearn
import time

app = Flask(__name__)

# Load the trained sentiment analysis model
loaded_LR_model = pickle.load(open(r"models/logistic_model.sav", "rb"))
loaded_vectorizer = pickle.load(open(r"models/Countvectorizer.sav", "rb"))

# Define a function to predict the sentiment
def predict(text):
    # Vectorize using bow vectorizer
    text_vector = loaded_vectorizer.transform([text])

    # Make predictions using the loaded model
    start_time = time.time()
    sentiment_prediction = loaded_LR_model.predict(text_vector)
    end_time = time.time()
    prediction_time = end_time - start_time

    return sentiment_prediction[0], prediction_time

# Define routes and render HTML templates
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.strip() != '':
            sentiment, prediction_time = predict(user_input)
            return render_template("result.html", sentiment=sentiment, prediction_time=prediction_time)
        else:
            return render_template("result.html", error="Please enter some text.")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')