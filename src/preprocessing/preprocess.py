import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)
    print("✅ Data loaded")
    return df


def preprocess_data(df):
    print("🔧 Starting preprocessing...")

    # Select important columns
    columns = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "tempo",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness"
    ]

    df = df[columns]

    # Drop missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    print("✅ Preprocessing complete")
    return df


def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved processed data to {path}")


if __name__ == "__main__":
    print(os.getcwd())
    raw_path = "data/raw/spotify_tracks.csv"
    processed_path = "data/processed/cleaned_songs.csv"

    df = load_data(raw_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, processed_path)

    print("🎉 Data preprocessing finished!")
    print(f"Final dataset shape: {df_clean.shape}")

    