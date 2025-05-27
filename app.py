import streamlit as st
import pandas as pd
import joblib
import os
from utils.predictor import predict

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")

available_models = {
    "Logistic Regression": "logistic_regression_heart_disease_model.joblib",
    "Random Forest": "random_forest_heart_disease_model.joblib",
    "Gradient Boosting": "gradient_boosting_heart_disease_model.joblib",
    "XGBoost": "xgboost_heart_disease_model.joblib"
}

selected_model_name = st.selectbox("🧠 Choose a model for prediction", list(available_models.keys()))

# Add explanation for each model here
model_descriptions = {
    "Logistic Regression": (
        "🔍 **Logistic Regression**: A simple and interpretable model.\n"
        "- Accuracy: ~78%\n"
        "- Great at spotting healthy individuals (82% precision)\n"
        "- Struggles a bit with those at risk (62% recall)\n"
        "👉 Best if you want something fast and basic."
    ),
    "Random Forest": (
        "🌲 **Random Forest**: Reliable and balanced.\n"
        "- Accuracy: 90%\n"
        "- High precision (89%) and solid recall (78%) on positives\n"
        "👉 Best all-rounder — robust and trustworthy."
    ),
    "Gradient Boosting": (
        "⚡ **Gradient Boosting**: Almost Random Forest, but with a twist.\n"
        "- Accuracy: 89%\n"
        "- Excellent precision (89%), slightly weaker recall (75%)\n"
        "👉 Good choice when precision matters more than catching every case."
    ),
    "XGBoost": (
        "🚀 **XGBoost**: Top dog for precision.\n"
        "- Accuracy: 90%\n"
        "- Highest precision on at-risk patients (93%)\n"
        "- Slightly lower recall (75%)\n"
        "👉 Best if you want fewer false alarms, and can tolerate a few misses."
    ),
}

# Show model explanation
st.markdown(f"**About {selected_model_name}:**")
st.info(model_descriptions[selected_model_name])

# Load the selected model
model_path = os.path.join("model", available_models[selected_model_name])
model = joblib.load(model_path)

def user_input_features():
    # Create mapping dictionaries for user-friendly labels -> model numeric codes
    yes_no_map = {"No": 0, "Yes": 1}
    sex_map = {"Female": 0, "Male": 1}

    # Use these mappings in the selectboxes and then convert to numeric
    features = {
        'HighBP': yes_no_map[st.selectbox('High Blood Pressure', options=list(yes_no_map.keys()))],
        'HighChol': yes_no_map[st.selectbox('High Cholesterol', options=list(yes_no_map.keys()))],
        'CholCheck': yes_no_map[st.selectbox('Cholesterol Check', options=list(yes_no_map.keys()))],
        'BMI': st.number_input('BMI', 10.0, 60.0, step=0.1),
        'Smoker': yes_no_map[st.selectbox('Smoker', options=list(yes_no_map.keys()))],
        'Stroke': yes_no_map[st.selectbox('Stroke', options=list(yes_no_map.keys()))],
        'Diabetes': yes_no_map[st.selectbox('Diabetes', options=list(yes_no_map.keys()))],
        'PhysActivity': yes_no_map[st.selectbox('Physical Activity', options=list(yes_no_map.keys()))],
        'Fruits': yes_no_map[st.selectbox('Eats Fruits', options=list(yes_no_map.keys()))],
        'Veggies': yes_no_map[st.selectbox('Eats Vegetables', options=list(yes_no_map.keys()))],
        'HvyAlcoholConsump': yes_no_map[st.selectbox('Heavy Alcohol Consumption', options=list(yes_no_map.keys()))],
        'AnyHealthcare': yes_no_map[st.selectbox('Access to Healthcare', options=list(yes_no_map.keys()))],
        'NoDocbcCost': yes_no_map[st.selectbox('No Doctor Due to Cost', options=list(yes_no_map.keys()))],
        'GenHlth': st.slider('General Health (1=Excellent, 5=Poor)', 1, 5),
        'MentHlth': st.slider('Mental Health Days (past 30)', 0, 30),
        'PhysHlth': st.slider('Physical Health Days (past 30)', 0, 30),
        'DiffWalk': yes_no_map[st.selectbox('Difficulty Walking', options=list(yes_no_map.keys()))],
        'Sex': sex_map[st.selectbox('Sex', options=list(sex_map.keys()))],
        'Age': st.slider('Age Group (coded)', 1, 13),
    }
    return pd.DataFrame([features])

input_df = user_input_features()

if st.button("Predict"):
    label, probability = predict(model, input_df)
    
    st.subheader("🩺 Prediction Result:")
    st.markdown(f"**Model Used**: `{selected_model_name}`")
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence Level:** {probability}%")

    if label == "Heart Disease":
        st.error("⚠️ Likely to have heart disease.")
    else:
        st.success("✅ Unlikely to have heart disease.")
