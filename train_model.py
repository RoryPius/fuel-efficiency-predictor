import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import json
import matplotlib.pyplot as plt
import shlex

# ── LOAD AND CLEAN DATA ──────────────────────────────────────────────────────
rows = []
with open('data/auto-mpg.data', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            parts = shlex.split(line)
            if len(parts) >= 8:
                rows.append(parts[:8])
        except:
            continue

columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
           'weight', 'acceleration', 'model_year', 'origin']

df = pd.DataFrame(rows, columns=columns)

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ── SPLIT FEATURES AND TARGET ──────────────────────────────────────────────
X = df.drop('mpg', axis=1)
y = df['mpg']


print(f" features shape: {X.shape}")
print(f" target shape: {y.shape}")
print(f" feature columns: {list(X.columns)}")

# ── TRAIN TEST SPLIT ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# ── TRAIN ALL THREE MODELS ───────────────────────────────────────────────────
models = {
    'Linear Regression':  LinearRegression(),
    'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    cv = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    results[name] = {'r2': r2, 'mae': mae, 'cv': cv}
    print(f"\n{name}")
    print(f"  R2 Score:       {r2:.4f}")
    print(f"  Avg Error:      {mae:.4f} mpg")
    print(f"  Cross-Val R2:   {cv:.4f}")

# ── SAVE THE BEST MODEL ──────────────────────────────────────────────────────
best_model = models['Random Forest']
joblib.dump(best_model, 'model/model.pkl')

feature_cols = list(X.columns)
with open('model/features.json', 'w') as f:
    json.dump(feature_cols, f)

print("\nModel saved to model/model.pkl")
print("Features saved to model/features.json")

# ── FEATURE IMPORTANCES ──────────────────────────────────────────────────────
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

importances.plot(kind='barh', color='#00d4ff', figsize=(8, 5))
plt.title('Feature Importances — What the Model Relies On')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('model/feature_importances.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFeature importances:")
for feature, importance in importances.sort_values(ascending=False).items():
    print(f"  {feature:15} {importance:.4f}")
