import pickle
import pandas as pd

MODEL_PATH = '../models/efficiency_pipeline.pkl'

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def engineer_features(df):
    df = df.copy()

    df["range_efficiency"] = df["range_km"] / (df["battery_kwh"] + 1e-3)
    df["speed_efficiency"] = df["range_km"] / (df["top_speed_kmph"] + 1e-3)
    df["range_per_seat"] = df["range_km"] / (df["seats"] + 1e-3)

    return df


def predict(vehicle_data: dict) -> dict:
    df = pd.DataFrame([vehicle_data])
    df = engineer_features(df)

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    return {
        'prediction': int(prediction),
        'prediction_label': 'High Efficiency' if prediction == 1 else 'Low Efficiency',
        'confidence': float(max(probabilities)),
        'probability_low': float(probabilities[0]),
        'probability_high': float(probabilities[1])
    }

if __name__ == '__main__':
    vehicle = {
        'battery_kwh': 75.0,
        'range_km': 500.0,
        'charging_time_hr': 1.0,
        'fast_charging': 1,
        'release_year': 2024,
        'seats': 5,
        'price_usd': 45000,
        'acceleration_0_100_kmph': 3.3,
        'top_speed_kmph': 225,
        'warranty_years': 4,
        'cargo_space_liters': 425,
        'safety_rating': 5.0,
        'type': 'Sedan',
        'drive_type': 'AWD',
        'fuel_type': 'Electric',
        'country': 'USA'
    }

    res = predict(vehicle)
    print(f"Prediction: {res['prediction_label']}")
    print(f"Confidence: {res['confidence']:.1%}")
    print(f"Probabilities: Low={res['probability_low']:.1%}, High={res['probability_high']:.1%}")