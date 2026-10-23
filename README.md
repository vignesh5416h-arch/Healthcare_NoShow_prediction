Healthcare Appointment No-Show Prediction

A Machine Learning project that analyzes healthcare appointment data and predicts potential patient no-shows using a Decision Tree Classifier. Power BI is used to visualize appointment attendance patterns and key performance indicators.

Project Overview

Missed healthcare appointments can affect scheduling and resource utilization. This project uses historical appointment data to explore attendance patterns and build a machine learning prediction workflow.

Objectives

- Clean and preprocess appointment data.
- Train a Decision Tree classification model.
- Generate predictions using Python.
- Analyze actual and predicted appointment outcomes.
- Visualize attendance patterns using Power BI.
- Present insights through an interactive dashboard.

Technologies Used

- Python — Programming and prediction workflow
- pandas — Data cleaning and manipulation
- scikit-learn — Machine learning
- joblib — Saving and loading the trained model
- Power BI — Data visualization and dashboard development
- VS Code — Development environment

Project Structure

Healthcare-NoShow-Prediction/
│
├── data/
│   └── cleaned_dataset.csv
│
├── outputs/
│   ├── models/
│   │   └── no_show_model.joblib
│   └── prediction_results.csv
│
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
├── Healthcare_NoShow_Dashboard.pbix
└── Healthcare_Appointment_No_Show_Prediction_Report.pdf

Note: This is a suggested structure. Keep the actual filenames from your project and include only files that exist.

Dataset Summary

The Power BI dashboard reports the following results:

Metric| Value
Total Appointments| 110,521
No-Show Appointments| 22,314
Show-Up Appointments| 88,207
No-Show Rate| 20.19%

These values describe the dataset shown in the dashboard; they do not represent model accuracy.

Machine Learning Workflow

1. Inspect and clean the dataset.
2. Prepare the features for machine learning.
3. Train a Decision Tree classifier.
4. Save the trained model using joblib.
5. Generate predictions using "predict.py".
6. Load the relevant results into Power BI.
7. Create KPI cards and analytical charts.
8. Review the results and document the project.

Power BI Dashboard

The dashboard includes four KPI cards and four analytical charts:

- Total appointments
- No-show appointments
- Show-up appointments
- No-show rate
- Age-wise analysis
- Gender-wise analysis
- SMS reminder analysis
- Weekday or appointment-date analysis

How to Run

1. Install a compatible version of Python.

2. Clone or download this repository.

3. Open the project folder in VS Code.

4. Install the required packages:
   
   pip install -r requirements.txt

5. Ensure the dataset and saved model are available at the paths expected by the code.

6. Run the prediction script:
   
   python predict.py

Important: The script may require a specific input CSV or feature columns. Check "predict.py" and follow its actual input requirements before running it.

Future Improvements

- Evaluate the model using precision, recall, F1-score, and a confusion matrix.
- Compare Decision Tree performance with other classification models.
- Improve the Power BI dashboard with model evaluation visuals.
- Investigate model performance across relevant patient groups.
- Explore responsible appointment reminder strategies.

Author

Aleti Vignesh
B.Tech — Computer Science and Engineering (AI & ML)

Disclaimer

This project is intended for educational and analytical purposes. Predictions should not replace professional judgment or be used to make decisions that unfairly disadvantage patients.
