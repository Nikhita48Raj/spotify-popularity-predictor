import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/best_model.pkl")

st.title("🎧 Spotify Song Popularity Predictor")

st.write("Adjust song features to predict popularity")

# Sliders for input
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

# Predict button
if st.button("Predict Popularity"):
    features = np.array([[danceability, energy, loudness, tempo,
                          valence, acousticness, speechiness, instrumentalness]])

    prediction = model.predict(features)
    st.progress(min(int(prediction[0]), 100))

    st.success(f"🎯 Predicted Popularity: {prediction[0]:.2f}")