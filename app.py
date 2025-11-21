from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model_clean.pkl")

@app.get("/")
def home():
    return {"status": "running on Fly.io"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}
