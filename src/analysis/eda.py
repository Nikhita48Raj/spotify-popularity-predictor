import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set_style("whitegrid")

# Load dataset
df = pd.read_csv("data/processed/cleaned_songs.csv")

# Basic info
print(df.info())
print(df.describe())

# Popularity distribution
plt.figure()
sns.histplot(df["popularity"], bins=30)
plt.title("Popularity Distribution")
plt.show()

# Feature vs popularity plots
features = ["danceability", "energy", "tempo", "valence"]

for feature in features:
    plt.figure()
    sns.scatterplot(x=df[feature], y=df["popularity"])
    plt.title(f"{feature} vs Popularity")
    plt.show()

# Histograms
df.hist(figsize=(12, 10))
plt.show()

# Boxplots for outlier detection
for col in df.columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()