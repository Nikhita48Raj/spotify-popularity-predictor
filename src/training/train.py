import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 📥 Load data
def load_data(path):
    df = pd.read_csv(path)
    print("✅ Data loaded")
    return df


# 🔀 Split data
def split_data(df):
    X = df.drop("popularity", axis=1)
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Data split into train and test")
    return X_train, X_test, y_train, y_test


# 🧠 Build ML pipeline
def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    print("✅ Pipeline created")
    return pipeline


# 📊 Evaluate model
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation Results:")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")


# 🔥 Feature Importance
def show_feature_importance(model, feature_names):
    print("\n📊 Feature Importance:")

    importances = model.named_steps["model"].feature_importances_

    for name, val in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {val:.3f}")


# 💾 Save model
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n💾 Model saved at {path}")


# 🚀 Main function
if __name__ == "__main__":
    data_path = "data/processed/cleaned_songs.csv"
    model_path = "models/model.pkl"

    # Load
    df = load_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Build
    model = build_pipeline()

    # Train
    print("\n🚀 Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    evaluate(model, X_test, y_test)

    # 🔥 SHOW FEATURE IMPORTANCE (NEW)
    show_feature_importance(model, X_train.columns)

    # Save
    save_model(model, model_path)

    print("\n🎉 Training complete!")