import streamlit as st
import pandas as pd
import joblib
import numpy as np

import joblib
import requests
import os

# Google Drive direct link
url = "https://drive.google.com/uc?id=1dhtcPk1KT6fsWy0jilKOB3B-YwuNuMqZ"
filename = "best_model.pkl"

# Download only if not already present
if not os.path.exists(filename):
    print("📥 Downloading model from Google Drive...")
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            f.write(r.content)
    print("✅ Model downloaded successfully.")

# Load the model
model = joblib.load(filename)


# Load model and encoders
model = joblib.load("best_model.pkl")
le_education = joblib.load("le_education.pkl")
le_occupation = joblib.load("le_occupation.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_gender = joblib.load("le_gender.pkl")
# Load any other encoders if needed

# Streamlit config
st.set_page_config(page_title="WageWise: Smart Salary Analyzer", page_icon="💼", layout="centered")
st.title("💼 WageWise: Smart Salary Analyzer")
st.markdown("Predict how much an employee earns based on their details. Gain quick insights into what factors drive salaries in the organization.")

# Sidebar
st.sidebar.header("Enter Employee Details")
st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=120)

# User inputs
# Sidebar inputs
st.sidebar.header("Enter Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", le_education.classes_)
occupation = st.sidebar.selectbox("Occupation", le_occupation.classes_)
workclass = st.sidebar.selectbox("Workclass", le_workclass.classes_)
gender = st.sidebar.selectbox("Gender", le_gender.classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Build input Dataframe (must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'workclass': [workclass],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### Input Summary")
st.table(input_df)

# Preprocess function
def preprocess_input(df):
    df = df.copy()
    df['education'] = le_education.transform(df['education'])
    df['occupation'] = le_occupation.transform(df['occupation'])
    df['workclass'] = le_workclass.transform(df['workclass'])
    df['gender'] = le_gender.transform(df['gender'])
    return df

# Prediction section
if st.button("🔮 Predict Salary Class"):
    try:
        processed = preprocess_input(input_df)
        prediction = model.predict(processed)
        probability = model.predict_proba(processed).max() * 100 if hasattr(model, 'predict_proba') else None

        if prediction[0] == '>50K':
            st.success("This employee is likely to earn **above $50K/year**.")
        else:
            st.warning("This employee is likely to earn **$50K/year or less**.")

        if probability:
            st.markdown(f"**Confidence Level:** {probability:.2f}%")

        st.balloons()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Batch Prediction Section
st.markdown("---")
st.markdown("### 📂 Batch Prediction for Multiple Employees")

uploaded_file = st.file_uploader("Upload a CSV file for salary prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:", batch_data.head())
    st.dataframe(batch_data.head())

    try:
        processed_data = preprocess_input(batch_data)
        batch_preds = model.predict(processed_data)
        batch_data['PredictedClass'] = batch_preds

        above_50k = (batch_preds == '>50K').sum()
        total = len(batch_preds)
        st.markdown(f"✅ **{above_50k}/{total} employees** predicted to earn >50K.")

        st.write("📊 Predictions with added column:")
        st.dataframe(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, file_name='salary_predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

st.markdown("---")
st.markdown("👩‍💻 *Built with ❤️ for HR professionals and data enthusiasts.*")
