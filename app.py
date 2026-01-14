from flask import Flask, request, jsonify
import joblib
import requests
import os

app = Flask(__name__)

# Load ML model
model = joblib.load("irrigation_model.pkl")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()

    # If API fails, use safe default values
    if "main" not in res:
        return 30.0, 70.0

    return float(res["main"]["temp"]), float(res["main"]["humidity"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print(data)

    soil = float(data["soil"])
    city = data["city"]

    temp, humidity = get_weather(city)

    # Get ML prediction
    water = float(model.predict([[soil, temp, humidity]])[0])

    return jsonify({
        "temperature": temp,
        "humidity": humidity,
        "water_needed": water,
        "irrigate": bool(water > 100)
    })


# Render requires this PORT binding
app.run(host="0.0.0.0", port=int(os.environ["PORT"]))
