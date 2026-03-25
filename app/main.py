from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/best_model.pkl")

# ✅ Define schema
class SongFeatures(BaseModel):
    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    loudness: float = Field(ge=-60, le=0)
    tempo: float = Field(ge=0, le=250)
    valence: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    speechiness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)


@app.post("/predict")
def predict(features: SongFeatures):
    data = np.array([[features.danceability, features.energy,
                      features.loudness, features.tempo,
                      features.valence, features.acousticness,
                      features.speechiness, features.instrumentalness]])

    prediction = model.predict(data)

    return {"predicted_popularity": float(prediction[0])}