import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
model_path = Path("outputs/models/no_show_model.joblib")
saved = joblib.load(model_path)

model = saved["model"]
features = saved["features"]

print("HEALTHCARE APPOINTMENT PREDICTION")
print("-" * 40)

# Get appointment details
gender = input("Enter gender (M/F): ").strip().upper()

if gender not in ["M", "F"]:
    raise ValueError("Gender must be M or F.")

age = int(input("Enter patient age: "))
scholarship = int(input("Scholarship (0=No, 1=Yes): "))
hypertension = int(input("Hypertension (0=No, 1=Yes): "))
diabetes = int(input("Diabetes (0=No, 1=Yes): "))
alcoholism = int(input("Alcoholism (0=No, 1=Yes): "))
handicap = int(input("Handicap level (0-4): "))
sms = int(input("SMS received (0=No, 1=Yes): "))
waiting_days = int(input("Days between scheduling and appointment: "))
weekday = int(input("Appointment weekday (0=Mon, 6=Sun): "))
month = int(input("Appointment month (1-12): "))

# Validate numeric inputs
if not 0 <= age <= 115:
    raise ValueError("Age must be between 0 and 115.")

if any(value not in [0, 1] for value in
       [scholarship, hypertension, diabetes, alcoholism, sms]):
    raise ValueError("Binary fields must be 0 or 1.")

if not 0 <= handicap <= 4:
    raise ValueError("Handicap level must be between 0 and 4.")

if waiting_days < 0:
    raise ValueError("Waiting days cannot be negative.")

if not 0 <= weekday <= 6 or not 1 <= month <= 12:
    raise ValueError("Invalid weekday or month.")

# Create input using the exact training feature names
row = {feature: 0 for feature in features}

values = {
    "Age": age,
    "Scholarship": scholarship,
    "Hipertension": hypertension,
    "Diabetes": diabetes,
    "Alcoholism": alcoholism,
    "Handcap": handicap,
    "SMS_received": sms,
    "WaitingDays": waiting_days,
    "AppointmentWeekday": weekday,
    "AppointmentMonth": month,
}

row.update(values)

# Match the gender encoding used during training
for feature in features:
    if feature.startswith("Gender_"):
        row[feature] = int(
            gender == feature.replace("Gender_", "")
        )

input_df = pd.DataFrame([row], columns=features)

# Predict
prediction = int(model.predict(input_df)[0])
probabilities = model.predict_proba(input_df)[0]
missed_probability = float(
    probabilities[list(model.classes_).index(1)]
)

print("\nPrediction:", "Likely to miss" if prediction == 1
      else "Likely to attend")
print(f"Model-estimated missed-appointment probability: "
      f"{missed_probability:.1%}")
print("\nThis is a practice model, not a medical decision tool.")