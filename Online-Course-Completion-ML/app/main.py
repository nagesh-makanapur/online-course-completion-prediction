from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import InferenceModel

app = FastAPI(title="Online Course Completion Prediction API")

# Load trained model
model = InferenceModel()


# ⚠️ Replace these fields with actual feature columns in your dataset
class StudentData(BaseModel):
    hours_studied: float
    assignments_completed: int
    quiz_score: float


@app.post("/predict")
def predict(data: StudentData):
    input_features = data.dict()
    prediction = model.predict(input_features)
    return prediction
