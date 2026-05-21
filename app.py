from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
model = joblib.load('model/model.pkl')
with open('model/features.json') as f:
    feature_cols = json.load(f)

# Fuel type data
fuel_data = {
    'petrol':   {'multiplier': 1.00, 'co2': 120},
    'diesel':   {'multiplier': 0.92, 'co2': 110},
    'ethanol':  {'multiplier': 1.08, 'co2': 70},
    'cng':      {'multiplier': 1.22, 'co2': 55},
    'hydrogen': {'multiplier': 1.35, 'co2': 0},
}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data[col] for col in feature_cols]
    base_mpg = model.predict([features])[0]
    fuel = data.get('fuel_type', 'petrol').lower()
    info = fuel_data.get(fuel, fuel_data['petrol'])
    final_mpg = round(base_mpg * info['multiplier'], 1)
    co2 = info['co2']
    lper100km = round(235.21 / final_mpg, 1)
    return jsonify({
        'mpg': final_mpg,
        'co2': co2,
        'lper100km': lper100km,
        'fuel_type': fuel
    })


@app.route('/')
def home():
    return 'Fuel Efficiency Predictor API is running'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
