import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Optional (install if needed)
# pip install xgboost
from xgboost import XGBRegressor


# Load data
def load_data(path):
    df = pd.read_csv(path)
    print("✅ Data loaded")
    return df


# Split data
def split_data(df):
    X = df.drop("popularity", axis=1)
    y = df["popularity"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Evaluate model
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"📊 MAE: {mae:.2f}")
    print(f"📊 R2 Score: {r2:.4f}")


# Main
if __name__ == "__main__":
    data_path = "data/processed/cleaned_songs.csv"
    model_path = "models/best_model.pkl"

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # 🔥 Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor())
    ])

    # 🔥 Hyperparameter space
    param_grid = [
        {
            "model": [RandomForestRegressor()],
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None]
        },
        {
            "model": [Ridge()],
            "model__alpha": [0.1, 1.0, 10.0]
        },
        {
            "model": [XGBRegressor(objective="reg:squarederror")],
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 6]
        }
    ]

    print("🚀 Starting hyperparameter tuning...")

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("✅ Best Model Found!")
    print(search.best_params_)

    best_model = search.best_estimator_

    print("📈 Evaluating best model...")
    evaluate(best_model, X_test, y_test)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, model_path)

    print(f"💾 Best model saved at {model_path}")