# 🎧 Spotify Song Popularity Predictor

An end-to-end Machine Learning project that predicts the popularity of a song based on its audio features using a deployed interactive web app.

---

## 🚀 Live Demo

👉 (https://spotify-popularity-predictor-x47h.onrender.com/)

---

## 📌 Project Overview

This project builds a complete ML pipeline to predict **Spotify song popularity** using features like:

- 🎵 Danceability  
- ⚡ Energy  
- 🔊 Loudness  
- 🕒 Tempo  
- 😊 Valence  
- 🎼 Acousticness  
- 🗣️ Speechiness  
- 🎹 Instrumentalness  

It covers the full lifecycle:  
➡️ Data Collection → Preprocessing → Model Training → Evaluation → Deployment

---

## 🧠 Machine Learning Pipeline

### 1. 📥 Data Ingestion
- Collected song data using Spotify API  
- Extracted audio features and metadata  

### 2. 🧹 Data Preprocessing
- Handled missing values  
- Selected relevant features  
- Cleaned and structured dataset  

### 3. 🤖 Model Training
Built pipeline using:
- `StandardScaler`  
- `LinearRegression` *(optimized for deployment speed)*  

Also experimented with:
- `RandomForestRegressor`  

### 4. 📊 Model Evaluation
Metrics used:
- MAE (Mean Absolute Error)  
- R² Score  

### 5. 🌐 Deployment
- Built UI using **Streamlit**  
- Deployed on **Render**  
- Auto-trains model if not available  

---

## 🛠️ Tech Stack

- 🐍 Python  
- 📊 Pandas, NumPy  
- 🤖 Scikit-learn  
- 🌐 Streamlit  
- 💾 Joblib  
- 🚀 Render (Deployment)  

---

## 📂 Project Structure

```
spotify-popularity-ml/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   └── training/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
├── requirements.txt
└── README.md
```

---



## 🎮 How to Use

1. Open the app  
2. Adjust song feature sliders  
3. Click **Predict Popularity**  
4. View predicted score  

---

## 📊 Sample Output

- 🎯 Predicted Popularity Score (0–100)  
- 📈 Visual progress indicator  
- 🔥 Popularity category (Low / Medium / High)  

---

## 💡 Key Highlights

- ✅ End-to-end ML pipeline  
- ✅ Real-world dataset (Spotify API)  
- ✅ Production-ready deployment  
- ✅ Auto model training logic  
- ✅ Interactive UI  

---

## 🚀 Future Improvements

- ⚡ Use advanced models (LightGBM)  
- 🔍 Add real-time Spotify song search  
- 🐳 Deploy using Docker  
- 🔐 Add user authentication  

---

## 📌 Final Note

This project demonstrates a complete **end-to-end Machine Learning workflow**, from data ingestion to production deployment, with a focus on simplicity, scalability, and real-world applicability.

---

---
