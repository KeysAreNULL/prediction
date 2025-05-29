import streamlit as st
import pandas as pd
from utils.predictor import ensemble_predict

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")

def user_input_features():
    yes_no_map = {"No": 0, "Yes": 1}
    sex_map = {"Female": 0, "Male": 1}

    features = {
        'HighBP': yes_no_map[st.selectbox('High Blood Pressure', options=list(yes_no_map.keys()))],
        'HighChol': yes_no_map[st.selectbox('High Cholesterol', options=list(yes_no_map.keys()))],
        'CholCheck': yes_no_map[st.selectbox('Had Cholesterol Check in Past 5 Years', options=list(yes_no_map.keys()))],
        'BMI': st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=20.0, step=0.1),
        'Smoker': yes_no_map[st.selectbox('Has Smoked at Least 100 Cigarettes over Lifetime', options=list(yes_no_map.keys()))],
        'Stroke': yes_no_map[st.selectbox('Ever Had a Stroke', options=list(yes_no_map.keys()))],
        'Diabetes': yes_no_map[st.selectbox('Has Diabetes', options=list(yes_no_map.keys()))],
        'PhysActivity': yes_no_map[st.selectbox('Physical Activity in Past 30 Days (excluding job)', options=list(yes_no_map.keys()))],
        'Fruits': yes_no_map[st.selectbox('Consumes Fruits Daily', options=list(yes_no_map.keys()))],
        'Veggies': yes_no_map[st.selectbox('Consumes Vegetables Daily', options=list(yes_no_map.keys()))],
        'HvyAlcoholConsump': yes_no_map[st.selectbox('Heavy Alcohol Consumption', options=list(yes_no_map.keys()))],
        'AnyHealthcare': yes_no_map[st.selectbox('Has Any Form of Healthcare Coverage', options=list(yes_no_map.keys()))],
        'NoDocbcCost': yes_no_map[st.selectbox('Couldn’t See Doctor Due to Cost', options=list(yes_no_map.keys()))],
        'GenHlth': st.slider('General Health (1 = Excellent, 5 = Poor)', min_value=1, max_value=5, value=3),
        'MentHlth': st.slider('Days of Poor Mental Health in the Past 30 Days', min_value=0, max_value=30, value=0),
        'PhysHlth': st.slider('Days of Poor Physical Health in the Past 30 Days', min_value=0, max_value=30, value=0),
        'DiffWalk': yes_no_map[st.selectbox('Difficulty Walking or Climbing Stairs', options=list(yes_no_map.keys()))],
        'Sex': sex_map[st.selectbox('Sex', options=list(sex_map.keys()))],
        'Age': st.number_input('Age (in years)', min_value=0, max_value=120, value=30, step=1),
    }
    return pd.DataFrame([features])

input_df = user_input_features()

result = None  # Initialize to None, so no error on first run

if st.button("Predict"):
    result = ensemble_predict(input_df)
    
    st.subheader("🩺 Prediction Result:")
    st.markdown(result["details"])

    if result["label"] == "Conflicting":
        st.warning("⚠️ Conflicting predictions — results inconclusive. Please consult a real doctor.")
    else:
        if result["avg_confidence"] < 60:
            st.warning(f"⚠️ Low confidence prediction ({result['avg_confidence']}%). Please interpret cautiously.")
        if result["label"] == "Heart Disease":
            st.error("⚠️ Likely to have heart disease.")
        else:
            st.success("✅ Unlikely to have heart disease.")
    
    st.markdown(
        """
        ---
        ### 🧩 About the Models:
        - **Logistic Regression**: Good for simple, interpretable relationships.
        - **Random Forest**: Strong with complex, non-linear patterns.
        - **Gradient Boosting**: Excellent at fine-tuning predictions.
        - **XGBoost**: Powerful, optimized for accuracy and speed.

        ⚠️ **Disclaimer:** This prediction tool is for informational purposes only and should **NOT** be used as a final diagnosis. Always consult a qualified healthcare professional for medical advice.
        ---
        """
    )

if result is not None:
    with st.expander("🔍 View individual model results"):
        st.markdown("### 🧠 Model Predictions & Confidence Levels")

        predictions = result["raw"]["predictions"]
        confidences = result["raw"]["probabilities"]

        data = []
        for model_name in predictions:
            label = predictions[model_name]
            conf = confidences[model_name]
            emoji = "❤️" if label == "Heart Disease" else "💚"
            data.append({
                "Model": model_name,
                "Prediction": f"{emoji} {label}",
                "Confidence (%)": conf
            })

        df = pd.DataFrame(data)
        st.table(df)

        st.markdown(
            "⚠️ **Note:** Confidence percentages indicate how sure each model is about its prediction. "
            "If predictions differ, trust the majority but always consult a medical professional for an accurate diagnosis."
        )
