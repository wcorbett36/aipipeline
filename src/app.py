from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df).tolist()
    return {"prediction": prediction}