import pandas as pd
import numpy as np
import pickle


# loade the trained model

model = pickle.load(open("loan_status_model.pkl", "rb"))


def predict_loan_status(input_data):
    # Convert input data to dataframe
    input_df = pd.DataFrame(input_data, index=[0])
    # make preddiction
    prediction = model.predict(input_df)
    return prediction[0]


if __name__ == "__main__":
    # example input data
    input_data = {
        "Gender": 1,
        "Married": 0,
        "Dependents": 0,
        "Education": 1,
        "Self_Employed": 0,
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 200,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 1,
    }

    predicted_status = predict_loan_status(input_data)
    print(
        f"Predicted Loan Status: {'Approved' if predicted_status == 1 else 'Rejected'}"
    )
