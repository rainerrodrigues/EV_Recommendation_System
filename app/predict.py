import pickle
import pandas as pd
from app.preprocess import DataPreprocessor

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["range_efficiency"] = df["range_km"] / (df["battery_kwh"] + 1e-3)
    df["speed_efficiency"] = df["range_km"] / (df["top_speed_kmph"] + 1e-3)
    df["range_per_seat"] = df["range_km"] / (df["seats"] + 1e-3)

    return df


def predict(vehicle_data: dict) -> dict:
    df = pd.DataFrame([vehicle_data])
    df = engineer_features(df)

    X, _ = preprocessor.transform(df)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "prediction": int(pred),
        "prediction_label": "High Efficiency" if pred == 1 else "Low Efficiency",
        "confidence": float(max(probs)),
        "probability_low": float(probs[0]),
        "probability_high": float(probs[1]),
    }
