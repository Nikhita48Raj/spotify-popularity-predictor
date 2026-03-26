from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(title="Spotify Popularity Predictor API")

# Paths
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")


# Load model once
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None


# Request schema
class SongFeatures(BaseModel):
    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    loudness: float = Field(ge=-60, le=0)
    tempo: float = Field(ge=50, le=200)
    valence: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    speechiness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)


def get_feature_ranges():
    return {
        "danceability": (0.0, 1.0, 0.1),
        "energy": (0.0, 1.0, 0.1),
        "loudness": (-60.0, 0.0, 2.0),
        "tempo": (50.0, 200.0, 5.0),
        "valence": (0.0, 1.0, 0.1),
        "acousticness": (0.0, 1.0, 0.1),
        "speechiness": (0.0, 1.0, 0.05),
        "instrumentalness": (0.0, 1.0, 0.1),
    }


def get_category(score: float) -> str:
    if score < 40:
        return "Low popularity"
    elif score < 70:
        return "Medium popularity"
    return "High popularity"


def suggest_improvements(model, features):
    feature_names = [
        "danceability",
        "energy",
        "loudness",
        "tempo",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
    ]

    ranges = get_feature_ranges()
    base_pred = model.predict(features)[0]
    suggestions = []

    for i, feature_name in enumerate(feature_names):
        min_val, max_val, step = ranges[feature_name]
        current_value = features[0][i]

        # Try increasing
        increased = features.copy()
        increased[0][i] = min(current_value + step, max_val)
        increased_pred = model.predict(increased)[0]

        # Try decreasing
        decreased = features.copy()
        decreased[0][i] = max(current_value - step, min_val)
        decreased_pred = model.predict(decreased)[0]

        best_direction = None
        best_value = current_value
        best_pred = base_pred

        if increased_pred > best_pred:
            best_direction = "increase"
            best_value = increased[0][i]
            best_pred = increased_pred

        if decreased_pred > best_pred:
            best_direction = "decrease"
            best_value = decreased[0][i]
            best_pred = decreased_pred

        improvement = best_pred - base_pred

        if best_direction is not None and improvement > 0.5:
            suggestions.append({
                "feature": feature_name,
                "direction": best_direction,
                "new_value": round(float(best_value), 2),
                "new_score": round(float(best_pred), 2),
                "improvement": round(float(improvement), 2)
            })

    suggestions.sort(key=lambda x: x["improvement"], reverse=True)
    return suggestions[:3]


@app.get("/")
def home():
    return {"message": "Spotify Popularity Predictor API is running"}


@app.post("/predict")
def predict(features: SongFeatures):
    if model is None:
        return {"error": "Model file not found. Please train the model first."}

    data = np.array([[
        features.danceability,
        features.energy,
        features.loudness,
        features.tempo,
        features.valence,
        features.acousticness,
        features.speechiness,
        features.instrumentalness
    ]])

    prediction = float(model.predict(data)[0])
    category = get_category(prediction)
    suggestions = suggest_improvements(model, data)

    return {
        "predicted_popularity": round(prediction, 2),
        "category": category,
        "suggestions": suggestions
    }