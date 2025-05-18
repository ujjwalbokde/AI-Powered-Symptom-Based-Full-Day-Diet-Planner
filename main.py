from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import json

app = FastAPI(title="Health Prediction API")

# Load Disease Prediction Model and Encoders
disease_model = tf.keras.models.load_model("model/disease_predictor_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    disease_label_encoder = pickle.load(f)
with open("model/symptom_list.json", "r") as f:
    all_symptoms = json.load(f)

# Load Diet Recommendation Model and Encoders
diet_model = tf.keras.models.load_model("model/diet_recommendation_model.h5")
with open("model/disease_encoder.pkl", "rb") as f:
    disease_encoder = pickle.load(f)
with open("model/gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)
with open("model/diet_encoder.pkl", "rb") as f:
    diet_encoder = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

### ✅ Request Model
class HealthInput(BaseModel):
    symptoms: list
    age: int
    bmi: float
    gender: str

### ✅ Root Endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Health Prediction API"}

### ✅ Combined Prediction Endpoint
@app.post("/predict-health")
def predict_health(input_data: HealthInput):
    try:
        # Disease Prediction
        input_vector = [1 if symptom in input_data.symptoms else 0 for symptom in all_symptoms]
        disease_features = np.array(input_vector).reshape(1, -1)

        disease_prediction = disease_model.predict(disease_features)
        predicted_disease_index = np.argmax(disease_prediction, axis=1)[0]
        predicted_disease = disease_label_encoder.inverse_transform([predicted_disease_index])[0]
        disease_confidence = float(np.max(disease_prediction))

        # Diet Recommendation
        encoded_disease = disease_encoder.transform([predicted_disease])[0]
        encoded_gender = gender_encoder.transform([input_data.gender])[0]
        scaled_features = scaler.transform([[encoded_disease, input_data.age, input_data.bmi, encoded_gender]])

        diet_prediction = diet_model.predict(scaled_features)
        predicted_diet_index = np.argmax(diet_prediction, axis=1)[0]
        predicted_diet = diet_encoder.inverse_transform([predicted_diet_index])[0]
        diet_confidence = float(np.max(diet_prediction))

        return {
            "predicted_disease": {
                "disease": predicted_disease,
                "confidence": round(disease_confidence, 4)
            },
            "recommended_diet": {
                "diet_type": predicted_diet,
                "confidence": round(diet_confidence, 4)
            }
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
