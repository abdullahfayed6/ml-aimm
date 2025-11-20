from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

with open("model_v2.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
