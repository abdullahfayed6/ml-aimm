from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

with open("model_v2.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"status": "Render ML API is running!"}

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
