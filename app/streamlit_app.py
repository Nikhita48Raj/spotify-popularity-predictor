import sys
import os

# ✅ Fix module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import numpy as np

# 📁 Paths
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_songs.csv")

# 🎧 UI Title
st.title("🎧 Spotify Song Popularity Predictor")
st.write("Adjust song features to predict popularity")


# 🚀 Load or Train Model (SAFE VERSION)
@st.cache_resource
def load_or_train_model():
    try:
        # ✅ Load if exists
        if os.path.exists(model_path):
            return joblib.load(model_path)

        # 🚀 Train if not exists
        with st.spinner("🚀 Training model for first time... please wait"):
            from src.training.train import load_data, split_data, build_pipeline

            df = load_data(data_path)
            X_train, X_test, y_train, y_test = split_data(df)

            model = build_pipeline()
            model.fit(X_train, y_train)

            os.makedirs(os.path.join(BASE_DIR, "..", "models"), exist_ok=True)
            joblib.dump(model, model_path)

        st.success("✅ Model ready!")
        return model

    except Exception as e:
        st.error(f"❌ Error while loading/training model: {e}")
        st.stop()


# Load model safely
model = load_or_train_model()


# 🎛️ Sliders
st.subheader("🎛️ Song Features")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)


# 🔮 Prediction
if st.button("Predict Popularity"):
    try:
        features = np.array([[danceability, energy, loudness, tempo,
                              valence, acousticness, speechiness, instrumentalness]])

        prediction = model.predict(features)[0]

        st.progress(min(int(prediction), 100))

        # 🎯 Output
        st.success(f"🎯 Predicted Popularity: {prediction:.2f}")

        # 🔥 Extra UX (optional)
        if prediction < 40:
            st.info("📉 Low popularity track")
        elif prediction < 70:
            st.warning("📊 Medium popularity track")
        else:
            st.success("🔥 High popularity track!")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")