import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score

DATA_PATH = "loan_approval_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def build_pipeline(data: pd.DataFrame):
    df = data.copy()

    df["Education_Level"] = df["Education_Level"].astype(str)
    df["Loan_Approved"] = df["Loan_Approved"].astype(str)

    le_education = LabelEncoder()
    df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

    cat_cols = [
        "Employment_Status",
        "Marital_Status",
        "Property_Area",
        "Loan_Purpose",
        "Employer_Category",
        "Gender",
    ]
    num_cols = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
    ]

    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="mean")

    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    encoded = ohe.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)

    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    df["DTI_Ratio_squared"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_squared"] = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log(df["Applicant_Income"] + 1)
    df = df.drop(columns=["DTI_Ratio", "Credit_Score"])

    feature_columns = [c for c in df.columns if c not in ["Applicant_ID", "Loan_Approved"]]
    X = df[feature_columns]
    y = df["Loan_Approved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return {
        "ohe": ohe,
        "scaler": scaler,
        "le_education": le_education,
        "le_target": le_target,
        "feature_columns": feature_columns,
        "X_scaled": X_scaled,
        "y": y,
    }


def transform_input(user_input: dict, ohe: OneHotEncoder, scaler: StandardScaler, le_education: LabelEncoder, feature_columns: list):
    df = pd.DataFrame([user_input])
    df["Education_Level"] = le_education.transform(df["Education_Level"].astype(str))

    cat_cols = [
        "Employment_Status",
        "Marital_Status",
        "Property_Area",
        "Loan_Purpose",
        "Employer_Category",
        "Gender",
    ]

    ohe_df = pd.DataFrame(
        ohe.transform(df[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index,
    )

    df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)
    df["DTI_Ratio_squared"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_squared"] = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log(df["Applicant_Income"] + 1)
    df = df.drop(columns=["DTI_Ratio", "Credit_Score"])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return scaler.transform(df)


def main():
    st.set_page_config(page_title="Credit Wise Loan Approval", layout="centered")
    st.title("Credit Wise Loan Approval Predictor")
    st.write("Enter borrower details and get a loan approval prediction from the trained model.")

    data = load_data()
    pipeline = build_pipeline(data)

    with st.sidebar:
        st.header("Applicant Features")
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=20000.0, step=100.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
        dependents = st.number_input("Dependents", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0, step=1.0)
        existing_loans = st.number_input("Existing Loans", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=5.0, value=0.35, step=0.01, format="%.2f")
        savings = st.number_input("Savings", min_value=0.0, value=5000.0, step=100.0)
        collateral = st.number_input("Collateral Value", min_value=0.0, value=15000.0, step=100.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=100.0)
        loan_term = st.number_input("Loan Term (months)", min_value=6.0, max_value=360.0, value=120.0, step=6.0)

        employment_status = st.selectbox("Employment Status", sorted(data["Employment_Status"].dropna().unique().tolist()))
        marital_status = st.selectbox("Marital Status", sorted(data["Marital_Status"].dropna().unique().tolist()))
        property_area = st.selectbox("Property Area", sorted(data["Property_Area"].dropna().unique().tolist()))
        loan_purpose = st.selectbox("Loan Purpose", sorted(data["Loan_Purpose"].dropna().unique().tolist()))
        education_level = st.selectbox("Education Level", sorted(data["Education_Level"].dropna().unique().tolist()))
        gender = st.selectbox("Gender", sorted(data["Gender"].dropna().unique().tolist()))
        employer_category = st.selectbox("Employer Category", sorted(data["Employer_Category"].dropna().unique().tolist()))

    user_input = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Property_Area": property_area,
        "Loan_Purpose": loan_purpose,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }

    X_scaled = pipeline["X_scaled"]
    y = pipeline["y"]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    accuracy = precision_score(y_test, ypred, average="weighted")

    if st.button("Predict Loan Approval"):
        features = transform_input(
            user_input,
            pipeline["ohe"],
            pipeline["scaler"],
            pipeline["le_education"],
            pipeline["feature_columns"],
        )
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        label = pipeline["le_target"].inverse_transform([prediction])[0]

        st.markdown("### Prediction")
        st.write(f"**Loan Approved:** {label}")
        st.write(f"**Approval Probability:** {proba[prediction] * 100:.1f}%")
        st.write(f"**Model accuracy on test split:** {accuracy * 100:.1f}%")

    st.markdown("---")
    st.write("### Dataset preview")
    st.dataframe(data.head(5))

if __name__ == "__main__":
    main()
