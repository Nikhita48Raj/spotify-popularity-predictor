import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Spotify Song Popularity Predictor",
    page_icon="🎧",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 1.5rem;
    }
    .section-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background-color: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        st.error("❌ Model not found. Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------
# Helper functions
# -----------------------------
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

def get_hit_ranges():
    return {
        "danceability": (0.60, 0.85),
        "energy": (0.65, 0.90),
        "loudness": (-10.0, -4.0),
        "tempo": (110.0, 140.0),
        "valence": (0.40, 0.70),
        "acousticness": (0.10, 0.40),
        "speechiness": (0.03, 0.12),
        "instrumentalness": (0.00, 0.10),
    }

def get_category(score):
    if score < 40:
        return "Low popularity track"
    elif score < 70:
        return "Medium popularity track"
    return "Potential HIT song"

def get_hit_probability(score):
    return max(0, min(score, 100))

def feature_feedback(feature_name, value):
    ideal_min, ideal_max = get_hit_ranges()[feature_name]
    if value < ideal_min:
        return "Increase"
    elif value > ideal_max:
        return "Decrease"
    return "Good"

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

        increased = features.copy()
        increased[0][i] = min(current_value + step, max_val)
        increased_pred = model.predict(increased)[0]

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
                "new_value": float(best_value),
                "new_score": float(best_pred),
                "improvement": float(improvement),
            })

    suggestions.sort(key=lambda x: x["improvement"], reverse=True)
    return suggestions[:3]

def create_radar_chart(feature_values):
    radar_features = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
    ]

    values = [feature_values[f] for f in radar_features]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.capitalize() for f in radar_features])

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

    ax.set_title("Song Feature Profile", pad=20)
    return fig

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🎧 Spotify Song Popularity Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict song popularity, estimate hit potential, and get suggestions to improve the track profile.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Input section
# -----------------------------
st.markdown("## 🎛️ Song Feature Inputs")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    tempo = st.slider("Tempo", 50.0, 200.0, 120.0)

with col2:
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)

predict_clicked = st.button("🚀 Predict Popularity", use_container_width=True)

# -----------------------------
# Prediction section
# -----------------------------
if predict_clicked:
    try:
        feature_values = {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "tempo": tempo,
            "valence": valence,
            "acousticness": acousticness,
            "speechiness": speechiness,
            "instrumentalness": instrumentalness,
        }

        features = np.array([[
            danceability,
            energy,
            loudness,
            tempo,
            valence,
            acousticness,
            speechiness,
            instrumentalness
        ]])

        prediction = float(model.predict(features)[0])
        hit_probability = get_hit_probability(prediction)
        category = get_category(prediction)
        suggestions = suggest_improvements(model, features)

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Popularity", f"{prediction:.2f}")
        m2.metric("Hit Probability", f"{hit_probability:.1f}%")
        m3.metric("Category", category)

        st.progress(min(max(int(prediction), 0), 100))

        if prediction < 40:
            st.info("📉 This track currently has low popularity potential.")
        elif prediction < 70:
            st.warning("📊 This track has medium popularity potential.")
        else:
            st.success("🔥 This track looks like a potential hit.")

        chart_col, feedback_col = st.columns([1.1, 1])

        with chart_col:
            st.markdown("### 🕸️ Song Profile Radar Chart")
            radar_fig = create_radar_chart(feature_values)
            st.pyplot(radar_fig)

        with feedback_col:
            st.markdown("### 📋 Feature Feedback")
            for feature_name, value in feature_values.items():
                feedback = feature_feedback(feature_name, value)
                st.write(f"**{feature_name.capitalize()}**: {feedback}")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("### 💡 Suggestions to Improve Popularity")
            if suggestions:
                for s in suggestions:
                    st.markdown(
                        f"""
                        <div class="section-card">
                        👉 <b>{s['direction'].capitalize()}</b> <b>{s['feature']}</b> to <b>{s['new_value']:.2f}</b><br>
                        Possible score: <b>{s['new_score']:.2f}</b> &nbsp; | &nbsp; Improvement: <b>+{s['improvement']:.2f}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.write("No strong improvement suggestions found for this input.")

        with right:
            st.markdown("### 🎧 Typical Ranges for Popular Songs")
            hit_ranges = get_hit_ranges()
            for feature_name, (low, high) in hit_ranges.items():
                st.write(f"**{feature_name.capitalize()}**: {low:.2f} to {high:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")