import streamlit as st
import joblib
import numpy as np
import os

# 📁 Paths
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_songs.csv")

# 🚀 Load or Train Model
if not os.path.exists(model_path):
    st.warning("⚠️ Model not found. Training model... (first run only)")

    from src.training.train import load_data, split_data, build_pipeline

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_pipeline()
    model.fit(X_train, y_train)

    os.makedirs(os.path.join(BASE_DIR, "..", "models"), exist_ok=True)
    joblib.dump(model, model_path)

    st.success("✅ Model trained successfully!")
else:
    model = joblib.load(model_path)


# 🎧 UI
st.title("🎧 Spotify Song Popularity Predictor")
st.write("Adjust song features to predict popularity")

st.subheader("🎛️ Song Features")

# Sliders
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

# Predict
if st.button("Predict Popularity"):
    features = np.array([[danceability, energy, loudness, tempo,
                          valence, acousticness, speechiness, instrumentalness]])

    prediction = model.predict(features)[0]

    # Progress bar
    st.progress(min(int(prediction), 100))

    # Output
    st.success(f"🎯 Predicted Popularity: {prediction:.2f}")