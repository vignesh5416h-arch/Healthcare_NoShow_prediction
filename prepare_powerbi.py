import pandas as pd
import joblib
from pathlib import Path

# Locate the cleaned dataset
possible_paths = [
    Path("data/cleaned/healthcare_appointments_cleaned.csv"),
    Path("data/healthcare_appointments_cleaned.csv")
]

data_path = next(
    (path for path in possible_paths if path.exists()), None
)

model_path = Path("outputs/models/no_show_model.joblib")
output_dir = Path("outputs/powerbi")
output_dir.mkdir(parents=True, exist_ok=True)

if data_path is None:
    raise FileNotFoundError(
        "Cleaned CSV not found. Check your data folder."
    )

if not model_path.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

# Load data and trained model
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

saved = joblib.load(model_path)
model = saved["model"]
features = saved["features"]

# Match the preprocessing used during training
for col in ["ScheduledDay", "AppointmentDay"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["WaitingDays"] = (
    df["AppointmentDay"].dt.normalize()
    - df["ScheduledDay"].dt.normalize()
).dt.days

df["AppointmentWeekday"] = df["AppointmentDay"].dt.dayofweek
df["AppointmentMonth"] = df["AppointmentDay"].dt.month

df["No-show"] = (
    df["No-show"].astype("string").str.strip()
    .map({"No": 0, "Yes": 1})
)

# Apply the training data filters
df = df[df["Age"].between(0, 115)].copy()
df = df[df["WaitingDays"] >= 0].copy()

required = [
    "No-show", "Age", "Gender", "Scholarship",
    "Hipertension", "Diabetes", "Alcoholism",
    "Handcap", "SMS_received", "WaitingDays",
    "AppointmentWeekday", "AppointmentMonth"
]
df = df.dropna(subset=required).copy()

# Encode gender, matching the saved training features
df["Gender"] = df["Gender"].astype(str).str.strip()
df = pd.get_dummies(df, columns=["Gender"], dtype=int)

X = df.reindex(columns=features, fill_value=0)

# Generate predictions and probabilities
df["Predicted_No_Show"] = model.predict(X)
classes = list(model.classes_)
missed_index = classes.index(1)

df["Predicted_No_Show_Probability"] = (
    model.predict_proba(X)[:, missed_index]
)

# Make the results readable
df["Actual_Outcome"] = df["No-show"].map({
    0: "Attended",
    1: "Missed"
})

df["Predicted_Outcome"] = df["Predicted_No_Show"].map({
    0: "Likely to attend",
    1: "Likely to miss"
})

df["Gender"] = df.apply(
    lambda row: (
        "M" if row.get("Gender_M", 0) == 1 else "F"
    ),
    axis=1
)

df["AppointmentWeekdayName"] = (
    df["AppointmentWeekday"].map({
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday",
        6: "Sunday"
    })
)

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[-1, 17, 29, 44, 59, 74, 115],
    labels=["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
)

# Export detailed prediction results
df.to_csv(output_dir / "prediction_results.csv", index=False)

# Export summary tables for Power BI
summaries = {
    "gender_summary.csv": "Gender",
    "sms_summary.csv": "SMS_received",
    "weekday_summary.csv": "AppointmentWeekdayName",
    "age_summary.csv": "AgeGroup"
}

for filename, group_column in summaries.items():
    summary = df.groupby(
        group_column, observed=True
    ).agg(
        Total_Appointments=("No-show", "size"),
        Actual_No_Shows=("No-show", "sum"),
        Predicted_No_Shows=("Predicted_No_Show", "sum")
    ).reset_index()

    summary.to_csv(output_dir / filename, index=False)

print("\nPower BI preparation completed!")
print("Records exported:", len(df))
print("Output folder:", output_dir)

for file in sorted(output_dir.glob("*.csv")):
    print("-", file.name)