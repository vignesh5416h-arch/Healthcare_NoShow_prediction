import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
input_file = Path("data/cleaned/healthcare_appointments_cleaned.csv")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(input_file)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 1. Attendance summary
print("\nNo-show counts:")
print(df["No-show"].value_counts())

print("\nNo-show percentages:")
print((df["No-show"].value_counts(normalize=True) * 100).round(2))

# 2. No-show distribution chart
df["No-show"].value_counts().reindex(["No", "Yes"]).plot(
    kind="bar", rot=0
)
plt.title("Appointment Attendance Distribution")
plt.xlabel("No-show (No = Attended, Yes = Missed)")
plt.ylabel("Number of Appointments")
plt.tight_layout()
plt.savefig(output_dir / "attendance_distribution.png")
plt.close()

# 3. Age distribution
df["Age"].hist(bins=30)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Appointments")
plt.tight_layout()
plt.savefig(output_dir / "age_distribution.png")
plt.close()

# 4. No-show rate by gender
gender_rate = (
    df.groupby("Gender")["No-show"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)
print("\nNo-show rate by gender (%):")
print(gender_rate.round(2))

gender_rate.plot(kind="bar", rot=0)
plt.title("No-show Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("No-show Rate (%)")
plt.tight_layout()
plt.savefig(output_dir / "no_show_by_gender.png")
plt.close()

# 5. SMS reminder comparison
sms_rate = (
    df.groupby("SMS_received")["No-show"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)
print("\nNo-show rate by SMS reminder status (%):")
print(sms_rate.round(2))

sms_rate.plot(kind="bar", rot=0)
plt.title("No-show Rate by SMS Reminder")
plt.xlabel("SMS Received (0 = No, 1 = Yes)")
plt.ylabel("No-show Rate (%)")
plt.tight_layout()
plt.savefig(output_dir / "no_show_by_sms.png")
plt.close()

print("\nEDA completed! Charts saved in the outputs folder.")