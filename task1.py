import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv(r"C:\Users\user\Downloads\archive (5)\Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Select useful columns
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Convert Male/Female into numbers
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])

# Features and Target
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 2))

# Detailed Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Sample Prediction
sample_passenger = [[1, 0, 25, 100]]
prediction = model.predict(sample_passenger)

if prediction[0] == 1:
    print("\nPassenger Predicted: Survived")
else:
    print("\nPassenger Predicted: Did Not Survive")
