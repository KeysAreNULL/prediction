def predict(model, input_data):
    """
    Predicts heart disease and returns both the class label and the probability.
    
    Args:
        model: Trained sklearn-like model with predict and predict_proba.
        input_data: 2D numpy array or DataFrame with features.
    
    Returns:
        tuple: (label_string, probability_percentage)
    """
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Probability of class 1 (heart disease)

    label = "Heart Disease" if pred == 1 else "No Heart Disease"
    prob_pct = round(prob * 100, 2)

    return label, prob_pct
