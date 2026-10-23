# Credit Risk Analysis System

## Project Overview
The Credit Risk Analysis System analyzes loan applicant data and predicts the possibility of loan default using Machine Learning.

## Objectives
- Analyze loan applicant data.
- Identify factors related to loan default.
- Compare three Machine Learning models.
- Generate graphs and reports using SQL.
- Predict the estimated default risk of a new applicant.

## Technologies Used
- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- SQL and SQLite
- Scikit-learn
- Gradio

## Dataset
The project uses a dataset containing 5,000 loan applicant records, including income, loan amount, employment length, loan purpose, and credit history.

## Project Features
1. Data cleaning and preprocessing
2. Exploratory Data Analysis with graphs
3. SQL-based risk summaries
4. Comparison of Logistic Regression, Decision Tree, and Random Forest
5. Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
6. New applicant default-risk prediction
7. Interactive dashboard

## Model Comparison

| Model | Accuracy | F1-score |
|---|---:|---:|
| Logistic Regression | 72.60% | 22.60% |
| Decision Tree | 60.40% | 38.13% |
| Random Forest | 70.40% | 22.11% |

The Decision Tree achieved the highest F1-score among the three models tested.

## Main Project Files
- `credit_risk_cleaned.csv` – Cleaned dataset
- `credit_risk.db` – SQLite database
- `model_comparison.csv` – Model evaluation results
- `credit_risk_model.pkl` – Saved machine learning model, if included
- `app.py` – Dashboard application

## How to Run
1. Open the project code in Google Colab.
2. Upload the required dataset and project files.
3. Run the code cells in order.
4. Open the dashboard to explore the analysis and test a new applicant.

## Conclusion
This project demonstrates how Python, SQL, data visualization, and Machine Learning can be used together to analyze credit risk and estimate loan default probability.

**Note:** The prediction is an estimate based on the trained model and should not be treated as the only basis for a real loan decision.
