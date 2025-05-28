import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
@app.route("/index")
def index():
    return
flask.render_template("index.html")
def home():
    return "ML Model API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["sepal_l"]),
            float(request.form["sepal_w"]),
            float(request.form["petal_l"]),
            float(request.form["petal_w"])
        ]

        features_array = np.array(features).reshape(1,-1)
        prediction = model.predict(features_array)[0]

        iris_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        predicted_class = iris_classes[prediction]

        return render_template("index.html", prediction=f"Iris Class: {predicted_class}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5010)
            
