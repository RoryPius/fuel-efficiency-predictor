# Fuel Efficiency Predictor
### PHINIA · AI Research Project · 2026

A full-stack machine learning application that predicts vehicle fuel efficiency, CO₂ emissions, and fuel consumption in real time. Built as an unsolicited research project by **Rory Pius** (BSc Computer Science – AI, Year 1) to demonstrate direct relevance to PHINIA's clean energy mission.

---

## What It Does

Enter six engine parameters, select a fuel type, and click **Run Prediction**. The system returns:

- **Predicted MPG** — miles per gallon, adjusted for fuel type
- **CO₂ Emissions** — grams per kilometre based on fuel
- **Litres / 100km** — European efficiency metric
- **PHINIA Insight** — a live message comparing your result to petrol baseline and PHINIA's Net Zero goals

Supports five fuel types: **Petrol · Diesel · Ethanol · CNG · Hydrogen**

---

## Project Structure

```
fuel-efficiency-predictor/
│
├── data/
│   └── auto-mpg.data          # UCI dataset — 392 vehicles (1970–1982)
│
├── model/
│   ├── model.pkl              # Trained Random Forest model (R² = 0.88)
│   ├── features.json          # Feature column order for predictions
│   └── feature_importances.png
│
├── notebook/
│   └── exploration.ipynb      # Data exploration and visualisations
│
├── app.py                     # Flask backend API
├── index.html                 # Frontend dashboard
├── train_model.py             # ML training script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## How to Run It Locally

### 1. Prerequisites

- Python 3.11+
- pip

### 2. Clone the Repository

```bash
git clone https://github.com/rorypius/fuel-efficiency-predictor.git
cd fuel-efficiency-predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Flask Backend

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

Keep this terminal open — the backend must be running for the frontend to work.

### 6. Open the Frontend

Open `index.html` in your browser directly (double-click the file, or drag it into Chrome/Firefox). No server needed for the frontend.

### 7. Run a Prediction

- Select a fuel type (Petrol, Diesel, Ethanol, CNG, or Hydrogen)
- Adjust the six engine sliders
- Click **Run Prediction**
- Results appear instantly in the right panel

---

## The Six Engine Inputs

| Input | Range | What It Represents |
|---|---|---|
| Cylinders | 3 – 8 | Number of engine cylinders |
| Displacement | 68 – 455 cu in | Engine volume — strongest predictor |
| Horsepower | 46 – 230 hp | Engine power output |
| Weight | 1500 – 5200 lbs | Vehicle weight |
| Acceleration | 8 – 25 s | 0–60 mph time |
| Model Year | 70 – 82 | Manufacturing year (technology proxy) |

---

## The Machine Learning Model

The model was trained on the [UCI Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg) — 392 real vehicles after cleaning.

Three algorithms were compared:

| Model | R² | MAE | Cross-Val R² |
|---|---|---|---|
| Linear Regression | 0.79 | 2.42 mpg | 0.59 |
| **Random Forest** | **0.88** | **1.72 mpg** | **0.76** |
| Gradient Boosting | 0.87 | 1.80 mpg | 0.75 |

**Random Forest was selected.** It explains 88% of fuel efficiency variation with an average error of just 1.72 mpg.

### Feature Importances

| Feature | Importance |
|---|---|
| Displacement | 41.6% |
| Horsepower | 16.9% |
| Weight | 14.4% |
| Cylinders | 13.8% |
| Model Year | 10.5% |
| Acceleration | 2.5% |
| Origin | 0.4% |

> Engine displacement is the single biggest predictor at 41.6% — the exact variable PHINIA's fuel injectors control.

---

## Fuel Type Logic

Each fuel type applies a multiplier to the base MPG prediction and carries a fixed CO₂ value:

| Fuel | MPG Multiplier | CO₂ (g/km) |
|---|---|---|
| Petrol | 1.00× (baseline) | 120 |
| Diesel | 0.92× | 110 |
| Ethanol | 1.08× | 70 |
| CNG | 1.22× | 55 |
| Hydrogen | 1.35× | 0 |

Switching from petrol to hydrogen increases predicted efficiency by **35%** and reduces direct CO₂ from **120 g/km to 0 g/km**.

---

## Retrain the Model (Optional)

If you want to retrain from scratch:

```bash
python train_model.py
```

This will:
1. Load and clean `data/auto-mpg.data`
2. Train all three models and print performance metrics
3. Save the best model to `model/model.pkl`
4. Save feature names to `model/features.json`
5. Generate and save the feature importances chart

---

## API Reference

The Flask backend exposes one endpoint:

### `POST /predict`

**Request body (JSON):**
```json
{
  "cylinders": 4,
  "displacement": 150,
  "horsepower": 100,
  "weight": 2800,
  "acceleration": 15,
  "model_year": 76,
  "origin": 1,
  "fuel_type": "hydrogen"
}
```

**Response (JSON):**
```json
{
  "mpg": 37.8,
  "co2": 0,
  "lper100km": 6.2,
  "fuel_type": "hydrogen"
}
```

---

## Version B — Planned Upgrade

The next version of this system will add a **File Upload Layer**: the ability to upload PDF, Word, or Excel files containing engine specification data. The system will extract the specs automatically and run batch predictions — enabling fleet-scale analysis without manual data entry.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | scikit-learn (Random Forest) |
| Backend API | Python · Flask · Flask-CORS |
| Model Serialisation | joblib |
| Data Processing | pandas · numpy |
| Frontend | HTML · CSS · JavaScript (no frameworks) |
| Font | Inter (Google Fonts) · JetBrains Mono |

---

## Requirements

```
flask>=3.0.0
flask-cors>=4.0.0
joblib>=1.3.0
numpy>=1.26.0
pandas>=2.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
```

---

## Troubleshooting

**"Connection error. Make sure app.py is running on port 5000."**
The Flask backend is not running. Go back to your terminal and run `python app.py` first.

**Port 5000 already in use**
```bash
# Mac — AirPlay Receiver uses 5000. Disable it in System Settings > General > AirDrop & Handoff
# Or change the port in app.py:
app.run(debug=True, port=5001)
# And update the fetch URL in index.html to match.
```

**Model not found error**
Make sure `model/model.pkl` exists. If not, run `python train_model.py` to generate it.

---

## Built By

**Rory Pius**
BSc Computer Science (Artificial Intelligence), University of kent, Year 1
[rorypius15@gmail.com]

Built entirely as an unsolicited research exercise to demonstrate genuine interest in PHINIA's mission and the technical capability to contribute to it from day one.

---
