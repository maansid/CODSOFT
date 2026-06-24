import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv(
    r"C:\Users\user\Downloads\archive (3)\IMDb Movies India.csv",
    encoding="latin1"
)

# Display first 5 rows
print(df.head())

# Remove rows with missing ratings
df = df.dropna(subset=["Rating"])

# Select required columns
df = df[["Genre", "Director", "Actor 1", "Rating"]]

# Fill missing values
df = df.fillna("Unknown")

# Convert text data into numbers
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor_encoder = LabelEncoder()

df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Director"] = director_encoder.fit_transform(df["Director"])
df["Actor 1"] = actor_encoder.fit_transform(df["Actor 1"])

# Features and Target
X = df[["Genre", "Director", "Actor 1"]]
y = df["Rating"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("\nModel Accuracy (R² Score):", round(score, 2))

# Sample Prediction
sample_movie = [[
    df["Genre"].iloc[0],
    df["Director"].iloc[0],
    df["Actor 1"].iloc[0]
]]

predicted_rating = model.predict(sample_movie)

print("Predicted Rating:", round(predicted_rating[0], 2))
