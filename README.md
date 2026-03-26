# 🎧 Spotify Song Popularity Predictor

An end-to-end Machine Learning project that predicts the popularity of a song based on its audio features using a deployed interactive web app with intelligent insights and visual analytics.

---

## 🚀 Live Demo

👉 [https://spotify-popularity-predictor-x47h.onrender.com/](https://spotify-popularity-predictor-x47h.onrender.com/)

---

## 📌 Project Overview

This project builds a complete ML pipeline to predict **Spotify song popularity** using features like:

* 🎵 Danceability
* ⚡ Energy
* 🔊 Loudness
* 🕒 Tempo
* 😊 Valence
* 🎼 Acousticness
* 🗣️ Speechiness
* 🎹 Instrumentalness

The application not only predicts popularity but also provides **data-driven insights** to help improve the likelihood of creating a popular track.

It covers the full lifecycle:
➡️ Data Collection → Preprocessing → Model Training → Evaluation → Deployment → Insight Generation

---

## 🧠 Machine Learning Pipeline

### 1. 📥 Data Ingestion

* Collected song data using Spotify API
* Extracted audio features and metadata

### 2. 🧹 Data Preprocessing

* Handled missing values
* Selected relevant numerical features
* Cleaned and structured dataset
* Prepared dataset for model training

### 3. 🤖 Model Training

Built pipeline using:

* `StandardScaler`
* `LinearRegression` *(optimized for faster prediction and deployment)*

Also experimented with:

* `RandomForestRegressor`

### 4. 📊 Model Evaluation

Metrics used:

* MAE (Mean Absolute Error)
* R² Score

Ensured model performance and generalization on unseen data.

### 5. 💡 Insight Generation

The app analyzes model predictions and provides:

* 🔥 Hit probability score
* 💡 Suggestions to improve popularity
* 📋 Feature-level feedback
* 🎧 Ideal feature ranges for successful songs

### 6. 🌐 Deployment

* Built UI using **Streamlit**
* Deployed on **Render**
* Automatically loads trained model
* Trains model automatically if not available

---

## ✨ App Features

### 🎛️ Interactive Feature Controls

Users can adjust song attributes using sliders:

* Danceability
* Energy
* Loudness
* Tempo
* Valence
* Acousticness
* Speechiness
* Instrumentalness

### 📊 Intelligent Output Dashboard

The app provides:

* 🎯 Predicted Popularity Score (0–100)
* 🔥 Hit Probability indicator
* 📈 Popularity category (Low / Medium / High)
* 🕸️ Radar chart visualization of song profile
* 💡 Suggestions to improve predicted popularity
* 📋 Feature-by-feature feedback
* 🎧 Ideal feature ranges observed in popular songs

---

## 🛠️ Tech Stack

* 🐍 Python
* 📊 Pandas, NumPy
* 🤖 Scikit-learn
* 📉 Matplotlib, Seaborn
* 🌐 Streamlit
* 💾 Joblib
* 🚀 Render (Deployment)

---

## 📂 Project Structure

```
spotify-popularity-ml/
│
├── app/
│   └── streamlit_app.py        # Streamlit web app
│
├── src/
│   ├── ingestion/              # data collection scripts
│   ├── preprocessing/          # data cleaning scripts
│   ├── training/               # ML pipeline scripts
│   └── analysis/               # EDA scripts
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/                  # experimentation notebooks
│
├── requirements.txt
└── README.md
```

---

## 🎮 How to Use

1. Open the deployed web app
2. Adjust song feature sliders
3. Click **Predict Popularity**
4. View insights including:

* Predicted popularity score
* Hit probability
* Radar chart visualization
* Feature feedback
* Suggestions for improvement

---

## 📊 Sample Output

* 🎯 Predicted Popularity Score (0–100)
* 🔥 Hit Probability indicator
* 📈 Popularity category (Low / Medium / High)
* 🕸️ Radar chart showing song profile
* 💡 Suggestions to improve track popularity

---

## 💡 Key Highlights

* ✅ End-to-end ML pipeline
* ✅ Real-world dataset using Spotify audio features
* ✅ Production-ready deployment
* ✅ Intelligent recommendation system
* ✅ Interactive and user-friendly UI
* ✅ Visual analytics using radar charts
* ✅ Actionable feature improvement suggestions

---

## 🚀 Future Improvements

* ⚡ Use advanced models (XGBoost / LightGBM)
* 🔍 Integrate real-time Spotify song search
* 🎼 Add genre prediction
* 📊 Compare multiple songs
* 🐳 Deploy using Docker
* 🔐 Add user authentication
* 📈 Add SHAP feature explainability

---

## 📌 Final Note

This project demonstrates a complete **end-to-end Machine Learning workflow**, from data ingestion to production deployment, enhanced with **visual insights and intelligent recommendations**.

It highlights the ability to design ML systems that not only make predictions but also provide meaningful, user-friendly insights.

---
⭐ Final Note

This project demonstrates how Machine Learning can be applied not only to generate predictions, but also to provide meaningful insights through an intuitive interface.

It showcases the ability to build scalable ML systems from data collection to deployment.
