import joblib
import pandas as pd


class InferenceModel:
    def __init__(self, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, features: dict):
        df = pd.DataFrame([features])
        return self.scaler.transform(df)

    def predict(self, features: dict):
        processed = self.preprocess(features)
        prediction = self.model.predict(processed)
        prob = self.model.predict_proba(processed)[0][1]
        return {
            "completed": int(prediction[0]),
            "completion_probability": round(float(prob), 4)
        }
