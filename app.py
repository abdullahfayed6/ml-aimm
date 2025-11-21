from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open("model_v2.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"status": "Fly.io ML API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
