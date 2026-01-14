from flask import Flask, request, jsonify
import joblib
import requests
import os

app = Flask(__name__)

model = joblib.load("irrigation_model.pkl")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()
    return res["main"]["temp"], res["main"]["humidity"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    soil = data["soil"]
    city = data["city"]

    temp, humidity = get_weather(city)

    water = model.predict([[soil, temp, humidity]])[0]

    return jsonify({
        "temperature": temp,
        "humidity": humidity,
        "water_needed": float(water),
        "irrigate": water > 100
    })

app.run(host="0.0.0.0", port=5000)
