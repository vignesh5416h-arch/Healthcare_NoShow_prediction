import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# File paths
input_file = Path("data/cleaned/healthcare_appointments_cleaned.csv")
model_dir = Path("outputs/models")
model_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# Remove rows with invalid ages
df = df[df["Age"].between(0, 115)].copy()

# Convert target: Yes = missed, No = attended
df["No-show"] = df["No-show"].map({"No": 0, "Yes": 1})

# Extract useful features from dates
df["ScheduledDay"] = pd.to_datetime(
    df["ScheduledDay"], errors="coerce"
)
df["AppointmentDay"] = pd.to_datetime(
    df["AppointmentDay"], errors="coerce"
)

df["WaitingDays"] = (
    df["AppointmentDay"].dt.normalize()
    - df["ScheduledDay"].dt.normalize()
).dt.days

df["AppointmentWeekday"] = df["AppointmentDay"].dt.dayofweek
df["AppointmentMonth"] = df["AppointmentDay"].dt.month

# Select features available in the dataset
feature_columns = [
    "Gender",
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "WaitingDays",
    "AppointmentWeekday",
    "AppointmentMonth"
]

# Keep valid records
df = df.dropna(
    subset=feature_columns + ["No-show"]
)

# Encode categorical gender
df = pd.get_dummies(
    df,
    columns=["Gender"],
    drop_first=True,
    dtype=int
)

feature_columns = [
    col for col in feature_columns if col != "Gender"
] + [
    col for col in df.columns if col.startswith("Gender_")
]

# Exclude negative waiting periods
df = df[df["WaitingDays"] >= 0]

X = df[feature_columns]
y = df["No-show"].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Attended", "Missed"]
))

# Save model and feature names
joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    model_dir / "no_show_model.joblib"
)

print("\nModel saved successfully!")
print("Location: outputs/models/no_show_model.joblib")