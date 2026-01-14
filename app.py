from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("irrigation_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    soil = data["soil"]
    temp = data["temp"]
    humidity = data["humidity"]

    water = model.predict([[soil, temp, humidity]])[0]

    return jsonify({
        "water_needed": float(water),
        "irrigate": water > 100
    })

app.run(host="0.0.0.0", port=5000)
