from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import json

app = FastAPI(title="Symptoms-Based Diet Recommendation API")

# Load Disease Prediction Model and its encoders
disease_model = tf.keras.models.load_model("model/disease_predictor_model.h5")
with open("model/label_encoder.pkl", "rb") as f:
    disease_label_encoder = pickle.load(f)
with open("model/symptom_list.json", "r") as f:
    all_symptoms = json.load(f)

# Load Diet Recommendation Model (NEW: uses only disease as input)
diet_model = tf.keras.models.load_model("model/disease_to_diet_model.h5")
with open("model/vectorizer.pkl", "rb") as f:
    disease_vectorizer = pickle.load(f)
with open("model/label_encoder_diet.pkl", "rb") as f:
    diet_label_encoder = pickle.load(f)

# Request Schema
class InputData(BaseModel):
    symptoms: list
    age: int
    bmi: float
    gender: str

# API root
@app.get("/")
def root():
    return {"message": "Welcome to the Symptoms-Based Diet Recommendation API"}

# Main endpoint
@app.post("/recommend-diet")
def recommend_diet(input_data: InputData):
    try:
        # Disease Prediction
        input_vector = [1 if symptom in input_data.symptoms else 0 for symptom in all_symptoms]
        disease_input = np.array(input_vector).reshape(1, -1)

        disease_probs = disease_model.predict(disease_input)
        predicted_disease_index = np.argmax(disease_probs)
        predicted_disease = disease_label_encoder.inverse_transform([predicted_disease_index])[0]
        disease_confidence = float(np.max(disease_probs))

        # Diet Recommendation (NEW model: disease -> diet)
        disease_vec = disease_vectorizer.transform([predicted_disease]).toarray()
        diet_probs = diet_model.predict(disease_vec)
        predicted_diet_index = np.argmax(diet_probs)
        predicted_diet = diet_label_encoder.inverse_transform([predicted_diet_index])[0]
        diet_confidence = float(np.max(diet_probs))

        return {
            "predicted_disease": {
                "disease": predicted_disease,
                "confidence": round(disease_confidence, 4)
            },
            "recommended_diet": {
                "diet_type": predicted_diet,
                "confidence": round(diet_confidence, 4)
            },
            "user_details": {
                "age": input_data.age,
                "bmi": input_data.bmi,
                "gender": input_data.gender
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")