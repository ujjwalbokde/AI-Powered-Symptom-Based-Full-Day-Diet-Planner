from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import json

app = FastAPI(title="Disease Prediction API")

# Load model and encoders
model = tf.keras.models.load_model("model/disease_predictor_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("model/symptom_list.json", "r") as f:
    all_symptoms = json.load(f)

class SymptomInput(BaseModel):
    symptoms: list

@app.get("/")
def root():
    return {"message": "Welcome to Disease Predictor API"}

@app.post("/predict")
def predict_disease(input_data: SymptomInput):
    try:
        input_vector = [1 if symptom in input_data.symptoms else 0 for symptom in all_symptoms]
        features = np.array(input_vector).reshape(1, -1)

        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction))

        return {
            "predicted_disease": predicted_disease,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
