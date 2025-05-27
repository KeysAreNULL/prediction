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
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)


def user_input_features():
    # Create mapping dictionaries for user-friendly labels -> model numeric codes
    yes_no_map = {"No": 0, "Yes": 1}
    sex_map = {"Female": 0, "Male": 1}

    # Use these mappings in the selectboxes and then convert to numeric
    features = {
        'HighBP': yes_no_map[st.selectbox('High Blood Pressure', options=list(yes_no_map.keys()))],
    'HighChol': yes_no_map[st.selectbox('High Cholesterol', options=list(yes_no_map.keys()))],
    'CholCheck': yes_no_map[st.selectbox('Had Cholesterol Check in Past 5 Years', options=list(yes_no_map.keys()))],
    'BMI': st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=20.0, step=0.1, help="Enter your BMI (e.g. 20.0 is normal weight)"),
    'Smoker': yes_no_map[st.selectbox('Has Smoked at Least 100 Cigarettes over Lifetime', options=list(yes_no_map.keys()))],
    'Stroke': yes_no_map[st.selectbox('Ever Had a Stroke', options=list(yes_no_map.keys()))],
    'Diabetes': yes_no_map[st.selectbox('Has Diabetes', options=list(yes_no_map.keys()))],
    'PhysActivity': yes_no_map[st.selectbox('Physical Activity in Past 30 Days (excluding job)', options=list(yes_no_map.keys()))],
    'Fruits': yes_no_map[st.selectbox('Consumes Fruits Daily', options=list(yes_no_map.keys()))],
    'Veggies': yes_no_map[st.selectbox('Consumes Vegetables Daily', options=list(yes_no_map.keys()))],
    'HvyAlcoholConsump': yes_no_map[st.selectbox('Heavy Alcohol Consumption', options=list(yes_no_map.keys()), help="Men: >14 drinks/week, Women: >7 drinks/week")],
    'AnyHealthcare': yes_no_map[st.selectbox('Has Any Form of Healthcare Coverage', options=list(yes_no_map.keys()))],
    'NoDocbcCost': yes_no_map[st.selectbox('Couldn’t See Doctor Due to Cost', options=list(yes_no_map.keys()))],
    'GenHlth': st.slider('General Health (1 = Excellent, 5 = Poor)', min_value=1, max_value=5, value=3),
    'MentHlth': st.slider('Days of Poor Mental Health in the Past 30 Days', min_value=0, max_value=30, value=0),
    'PhysHlth': st.slider('Days of Poor Physical Health in the Past 30 Days', min_value=0, max_value=30, value=0),
    'DiffWalk': yes_no_map[st.selectbox('Difficulty Walking or Climbing Stairs', options=list(yes_no_map.keys()))],
    'Sex': sex_map[st.selectbox('Sex', options=list(sex_map.keys()))],
    'Age': st.number_input('Age (in years)', min_value=0, max_value=120, value=30, step=1, help="Enter your actual age in whole years"),
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
