# Step 2: Write the Streamlit app code
# %%writefile app.py
import streamlit as st
import joblib
import pickle as pkl
import pandas as pd
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")

# 🧠 Context and background
st.markdown("""
Welcome to the **Heart Disease Prediction App**.  
This tool uses machine learning to estimate the likelihood that a patient has heart disease based on medical indicators.

**Prediction Levels (`num`)**:
- `0`: No heart disease
- `1`: Mild
- `2`: Moderate
- `3`: Severe

This app is trained using the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data), with over 900 patient records.

---
""")

# Inputs
age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)

origin = st.selectbox("Place of Study (origin)", options=[
    "Cleveland", "Hungary", "Switzerland", "VA Long Beach"
])

sex = st.selectbox("Sex", options=["Male", "Female"])

cp = st.selectbox("Chest Pain Type (cp) : Determined by doctor/lab", options=[
    "typical angina", "atypical angina", "non-anginal", "asymptomatic"
])

trestbps = st.number_input("Resting Blood Pressure (mm Hg) : Use a home BP monitor or clinic reading (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol : Blood test result(mg/dl)", min_value=100, max_value=600, value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs) : After 8–12 hours no eating. >120 mg/dL is high/Yes.", options=["Yes", "No"])

restecg = st.selectbox("Resting ECG Results", options=[
    "normal", "stt abnormality", "lv hypertrophy"
])


thalach = st.number_input("Max Heart Rate Achieved (thalach) : Measured during treadmill/stress test", min_value=60, max_value=250, value=150)

exang = st.selectbox("Exercise-Induced Angina (exang)", options=["True", "False"])

oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak) : From stress ECG test (by doctor)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)

st.markdown("[🩺 Learn More: Heart Tests & Diagnosis](https://www.heart.org/en/health-topics/heart-attack/diagnosing-a-heart-attack)")

# 33.59 missing percentage > 30% so removed column
# slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
#     "downsloping", "flat", "upsloping"
# ])

# 66.41 missing percentage > 30% so removed column
# ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])

# 52.82 missing percentage > 30% so removed column
# thal = st.selectbox("Thalassemia (thal)", options=[
#     "normal", "fixed defect", "reversible defect"
# ])

if st.button("Submit"):
    raw_data = {
        "id": 1,
        "age": age,
        "sex": sex,
        "dataset": origin,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalach,
        "exang": exang,
        "oldpeak": oldpeak
    }

    df = pd.DataFrame([raw_data])

    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    # Handle Outlier
    upper_bound_values = {
        'age': 74.0,
        'trestbps': 160.0,
        'chol': 363.5,
        'thalach': 194.0,
        'oldpeak': 3.0
    }

    lower_bound_values = {
        'age': 32.0,
        'trestbps': 100.0,
        'chol': 74.0,
        'thalach': 83.0,
        'oldpeak': -1.5
    }

    # Iterate through each numerical column and apply the outlier mask
    for col in numerical_columns:
        if col in upper_bound_values and col in lower_bound_values:
            df[col] = df[col].mask(
                (df[col] > upper_bound_values[col]) | (df[col] < lower_bound_values[col]),
                np.nan
            )

    # First Min Max Scaler
    URLmmscalerBeforeFeatureEngineering = "mmscalerBeforeFeatureEngineering.pkl"
    mmscalerBeforeFeatureEngineering = joblib.load(URLmmscalerBeforeFeatureEngineering)
    df[numerical_columns] = pd.DataFrame(mmscalerBeforeFeatureEngineering.transform(df[numerical_columns]), index=df.index, columns=df[numerical_columns].columns)

    # KNN Imputer
    URLknnImputer = "knnImputer.pkl"
    knnImputer = joblib.load(URLknnImputer)
    df[numerical_columns] = pd.DataFrame(knnImputer.transform(df[numerical_columns]), index=df.index, columns=df[numerical_columns].columns)

    # Feature Engineering
    df[numerical_columns] = mmscalerBeforeFeatureEngineering.inverse_transform(df[numerical_columns])

    df['age_group'] = pd.cut(
        df['age'], bins=[29, 40, 55, 70, 100],
        labels=['Young', 'Middle-aged', 'Senior', 'Elderly']
    )

    chol_bins = [0, 200, 240, np.inf]
    chol_labels = ['Desirable', 'Borderline High', 'High']

    df['chol_category'] = pd.cut(
        df['chol'],
        bins=chol_bins,
        labels=chol_labels,
        right=False
    )

    bp_bins = [0, 120, 130, 140, 180, np.inf]
    bp_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Crisis']

    df['bp_category'] = pd.cut(
        df['trestbps'],
        bins=bp_bins,
        labels=bp_labels,
        right=False
    )

    df['age_chol'] = df['age'] * df['chol']

    df['heart_rate_reserve'] = df['thalch'] / (220 - df['age'])

    df['thalch_div_age'] = df['thalch'] / (df['age'] + 1)
    df['peak_stress'] = df['oldpeak'] * df['thalch']

    # Second Min Max Scaler
    URLmmscalerAfterFeatureEngineering = "mmscalerAfterFeatureEngineering.pkl"
    mmscalerAfterFeatureEngineering = joblib.load(URLmmscalerAfterFeatureEngineering)
    df[numerical_columns] = pd.DataFrame(mmscalerAfterFeatureEngineering.transform(df[numerical_columns]), index=df.index, columns=df[numerical_columns].columns)

    # One Hot Encoding
    urlOneHotEncoder = "ohEncoder.pkl"
    loaded_ohe = joblib.load(urlOneHotEncoder)

    for col in categorical_columns:
      df[col] = df[col].astype(str)

    df = pd.DataFrame(loaded_ohe.transform(df), index=df.index, columns=loaded_ohe.get_feature_names_out())

    # Delete remainder__id
    df = df.drop(columns=['remainder__id'])


    # Manually get Feature Selection from AOL.ipynb
    columns_to_keep = ['ohe__dataset_Hungary', 'ohe__dataset_Switzerland',
                   'ohe__cp_asymptomatic', 'ohe__exang_False', 'ohe__exang_True',
                   'remainder__age', 'remainder__trestbps', 'remainder__chol',
                   'remainder__thalch', 'remainder__oldpeak', 'remainder__age_chol',
                   'remainder__heart_rate_reserve', 'remainder__thalch_div_age',
                   'remainder__peak_stress']

    df = df[columns_to_keep]

    # Pre-processed input
    st.subheader("Preprocessed Input for Model:")
    st.write(df)

    # Modelling

    urlPKL = "FinalModelStacking.pkl"
    model = joblib.load(urlPKL)

    # Make prediction
    proba = model.predict_proba(df)[0]

    st.subheader("Prediction: Probability for Each Heart Disease Severity Level (num)")
    for i, p in enumerate(proba):
        st.write(f"Chance of num = {i}: **{p * 100:.2f}%**")

    st.success(f"Most likely prediction: num = {np.argmax(proba)}")




# Step 3: Run the Streamlit app in the background
# !streamlit run app.py &>/content/logs.txt &

# Step 4: Expose the app using LocalTunnel and get the public IP for the tunnel password
# import urllib
# public_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
# print(f"Tunnel password is: {public_ip}")

# Step 5: Expose the app using LocalTunnel
# !npx localtunnel --port 8501
