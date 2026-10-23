import streamlit as st
import pandas as pd
import joblib

st.title("Credit Risk Analysis System")
st.write("Analyze loan applicant risk")

model = joblib.load("credit_risk_model.pkl")
data = pd.read_csv("credit_risk_cleaned.csv")

st.subheader("Dataset Overview")
st.write("Total Applicants:", len(data))
st.dataframe(data.head())

st.subheader("New Applicant Prediction")

age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Annual Income", min_value=0, value=40000)
loan = st.number_input("Loan Amount", min_value=0, value=5000)
interest = st.number_input("Interest Rate", min_value=0.0, value=12.5)
employment = st.number_input("Employment Length", min_value=0.0, value=2.0)
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
purpose = st.selectbox(
    "Loan Purpose",
    ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE",
     "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)
default = st.selectbox("Previous Default", ["N", "Y"])
history = st.number_input("Credit History Length", min_value=0, value=3)

if st.button("Predict Risk"):
    applicant = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": employment,
        "loan_intent": purpose,
        "loan_amnt": loan,
        "loan_int_rate": interest,
        "loan_percent_income": loan / income if income > 0 else 0,
        "cb_person_default_on_file": default,
        "cb_person_cred_hist_length": history
    }])

    prediction = model.predict(applicant)[0]
    st.write("Predicted Loan Status:", prediction)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(applicant)[0][1]
        st.write("Estimated Default Probability:", round(probability * 100, 2), "%")
