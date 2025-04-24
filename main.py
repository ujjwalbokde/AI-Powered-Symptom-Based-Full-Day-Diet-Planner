from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import io

app = FastAPI()

# Load models
keras_model = tf.keras.models.load_model("models/best_model.h5")
joblib_model = joblib.load("models/model_joblib.pkl")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Model API is live!"}

@app.post("/predict/keras")
def predict_keras(data: InputData):
    input_array = np.array([data.features])
    prediction = keras_model.predict(input_array)
    return {"keras_prediction": prediction.tolist()}

@app.post("/predict/joblib")
def predict_joblib(data: InputData):
    input_array = np.array([data.features])
    prediction = joblib_model.predict(input_array)
    return {"joblib_prediction": prediction.tolist()}
