import pandas as pd
from pathlib import Path

# File locations
input_file = Path("data\\raw\\KaggleV2-May-2016.csv\\KaggleV2-May-2016.csv")
output_dir = Path("data/cleaned")
output_file = output_dir / "healthcare_appointments_cleaned.csv"

# Create output folder
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip()

# Remove duplicate records
before = len(df)
df = df.drop_duplicates()
print("Duplicate records removed:", before - len(df))

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Convert dates
for column in ["ScheduledDay", "AppointmentDay"]:
    df[column] = pd.to_datetime(df[column], errors="coerce")

# Remove rows with invalid dates
df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])

# Standardize text columns
for column in ["Gender", "No-show"]:
    df[column] = df[column].astype("string").str.strip()

# Validate target labels
df = df[df["No-show"].isin(["Yes", "No"])]

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Saved cleaned dataset to:", output_file)
print("\nNo-show distribution:")
print(df["No-show"].value_counts())

print("\nCleaning completed successfully!")