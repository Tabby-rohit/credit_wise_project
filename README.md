# Credit Wise Loan Approval

A Streamlit app that predicts loan approval using a credit dataset and a trained machine learning pipeline.

## Overview

This project demonstrates a loan approval prediction workflow with the following steps:

- load and clean `loan_approval_data.csv`
- handle missing values for categorical and numerical columns
- encode categorical variables using one-hot encoding and label encoding
- engineer new features such as squared ratios and log-transformed income
- scale numerical features with `StandardScaler`
- train a `GaussianNB` model for loan approval prediction
- expose a web form for user input and generate predictions in real time

## Features Used

The app collects these borrower features:

- `Applicant_Income`
- `Coapplicant_Income`
- `Age`
- `Dependents`
- `Credit_Score`
- `Existing_Loans`
- `DTI_Ratio`
- `Savings`
- `Collateral_Value`
- `Loan_Amount`
- `Loan_Term`
- `Employment_Status`
- `Marital_Status`
- `Property_Area`
- `Loan_Purpose`
- `Education_Level`
- `Gender`
- `Employer_Category`

## Deployed App

[Credit Wise Loan Approval · Streamlit](https://tabby-rohit-credit-wise-project-streamlit-app-affd0u.streamlit.app/)

## Local Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open the URL shown in the terminal to use the app locally.

## Repository Structure

- `streamlit_app.py` — main Streamlit application and preprocessing/model pipeline
- `loan_approval_data.csv` — dataset used to train the loan approval predictor
- `requirements.txt` — Python dependencies needed to run the app
- `README.md` — project overview and usage instructions

## Notes

- The app trains the model on startup using the provided dataset.
- Predictions are shown with the estimated probability and the model's test accuracy.
