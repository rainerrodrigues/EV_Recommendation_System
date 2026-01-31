import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


MODEL_PATH = 'models/best_model_20260128_131331.pkl'
PREPROCESSOR_PATH = 'models/preprocessor_20260128_131331.pkl'

class DataPreprocessor:
    
    def __init__(self, numerical_features=None, categorical_features=None, target_col='target_high_efficiency'):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        self.num_medians = {}
        self.cat_modes = {}
    
    def fit(self, train_df):
        for col in self.numerical_features:
            self.num_medians[col] = train_df[col].median()
        
        for col in self.categorical_features:
            self.cat_modes[col] = train_df[col].mode()[0]
        
        for col in self.categorical_features:
            le = LabelEncoder()
            train_col = train_df[col].fillna(self.cat_modes[col])
            le.fit(train_col.astype(str))
            self.label_encoders[col] = le
        
        train_X = self._prepare_features(train_df)
        self.scaler.fit(train_X[self.numerical_features])
        self.is_fitted = True
        return self
    
    def transform(self, df, split_name=''):
        X = self._prepare_features(df)
        X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X, df[self.target_col] if self.target_col in df.columns else None
    
    def _prepare_features(self, df):
        X = df[self.numerical_features + self.categorical_features].copy()
        
        for col in self.numerical_features:
            X[col].fillna(self.num_medians[col], inplace=True)
        
        for col in self.categorical_features:
            X[col].fillna(self.cat_modes[col], inplace=True)
        
        for col in self.categorical_features:
            X[col] = X[col].astype(str)
            mask = ~X[col].isin(self.label_encoders[col].classes_)
            if mask.any():
                X.loc[mask, col] = self.label_encoders[col].classes_[0]
            X[col] = self.label_encoders[col].transform(X[col])
        
        return X


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

def engineer_features(df):
    df = df.copy()
    df['range_efficiency'] = df['range_km'] / (df['battery_kwh'] + 0.001)
    df['battery_range_ratio'] = df['battery_kwh'] / (df['range_km'] + 0.001)
    df['charging_speed'] = df['battery_kwh'] / (df['charging_time_hr'] + 0.001)
    df['power_indicator'] = 100 / (df['acceleration_0_100_kmph'] + 0.001)
    df['speed_efficiency'] = df['range_km'] / (df['top_speed_kmph'] + 0.001)
    df['vehicle_age'] = 2026 - df['release_year']
    df['price_per_kwh'] = df['price_usd'] / (df['battery_kwh'] + 0.001)
    df['range_per_seat'] = df['range_km'] / (df['seats'] + 0.001)
    df['is_electric'] = (df['fuel_type'] == 'Electric').astype(int)
    df['has_fast_charging'] = df['fast_charging'].astype(int)

    return df

def predict(vehicle_data):
    df = pd.DataFrame([vehicle_data])
    df = engineer_features(df)
    X, _ = preprocessor.transform(df)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

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
