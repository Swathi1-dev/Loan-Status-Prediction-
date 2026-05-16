import pandas as pd
import streamlit as st
from predict import predict_loan_status


def main():

    st.title("Loan Status Prediction")
    st.write("Enter the details to predict loan status:")
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])

        Dependents = st.number_input("Dependents", min_value=0, max_value=4, step=1)
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.number_input("Loan Amount", min_value=0)
        Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
        Credit_History = st.selectbox("Credit History", [1, 0])
        Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    if st.button("Predict"):
        input_data = {
            "Gender": 1 if Gender == "Male" else 0,
            "Married": 1 if Married == "Yes" else 0,
            "Dependents": Dependents,
            "Education": 1 if Education == "Graduate" else 0,
            "Self_Employed": 1 if Self_Employed == "Yes" else 0,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": (
                2
                if Property_Area == "Urban"
                else 1
                if Property_Area == "Semiurban"
                else 0
            ),
        }
        result = predict_loan_status(input_data)
        if result == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")


if __name__ == "__main__":
    main()
