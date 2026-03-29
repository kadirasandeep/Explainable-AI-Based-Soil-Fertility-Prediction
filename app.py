import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np


# Load the trained model
model = joblib.load('soil_fertility_rf_model.pkl')


# Define feature names
feature_names = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']


# Nutrient recommendations
recommendations = {
    'N': 'Apply urea or composted manure to increase nitrogen.',
    'P': 'Use single superphosphate or bone meal to improve phosphorus.',
    'K': 'Apply muriate of potash (MOP) or wood ash for potassium.',
    'Zn': 'Apply zinc sulfate to increase zinc levels.',
    'Fe': 'Use ferrous sulfate or chelated iron.',
    'Cu': 'Apply copper sulfate or compost with copper.',
    'Mn': 'Use manganese sulfate or foliar sprays.',
    'B': 'Apply borax or boric acid.',
    'S': 'Use gypsum or elemental sulfur.',
    'OC': 'Incorporate organic compost or green manure.',
    'pH': 'Adjust with lime (to raise) or sulfur (to lower).',
    'EC': 'High EC? Flush with water. Low EC? Add balanced fertilizer.'
}


# Thresholds for nutrient deficiency
low_thresholds = {
    'N': 150, 'P': 15, 'K': 150, 'pH': 5.5, 'EC': 0.5, 'OC': 0.75,
    'S': 10, 'Zn': 0.5, 'Fe': 4.0, 'Cu': 0.5, 'Mn': 5, 'B': 0.5
}


# SHAP Explainer
explainer = shap.Explainer(model)


# Predict function
def predict_soil_fertility(features):
    df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(df)[0]
    return prediction, df


# Streamlit UI
st.title('🌱 Soil Fertility Prediction with Explainable AI (XAI)')


# User Inputs
features = []
for feature in feature_names:
    value = st.number_input(f'Enter {feature}', value=0.0)
    features.append(value)


# Predict Button
if st.button('Predict'):
    prediction, input_df = predict_soil_fertility(features)

    # Output Prediction
    fertility_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    st.markdown(f"### 🔍 Predicted Soil Fertility: **{fertility_labels[prediction]}**")

    # Bar Chart
    st.subheader("📊 Nutrient Values - Bar Chart")
    fig1, ax1 = plt.subplots()
    ax1.bar(feature_names, features, color='skyblue')
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Values")
    ax1.set_title("Soil Nutrient Levels")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Line Chart
    st.subheader("📈 Nutrient Trend - Line Chart")
    fig2, ax2 = plt.subplots()
    ax2.plot(feature_names, features, marker='o', linestyle='-', color='green')
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Values")
    ax2.set_title("Soil Nutrient Trend")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # SHAP Explanation (Force Plot)
    st.subheader("🧠 SHAP Explanation (Feature Impact)")

    shap_values = explainer(input_df)

    # For multi-class models, pick the predicted class
    predicted_class = prediction
    expected_val = explainer.expected_value[predicted_class]
    shap_val = shap_values.values[0, :, predicted_class]

    # Create SHAP force plot (pass feature names)
    shap_html = shap.plots.force(
        expected_val,
        shap_val,
        feature_names=feature_names
    )

    # Render in Streamlit
    st.components.v1.html(shap.getjs() + shap_html.html(), height=300)

    # Recommendations
    st.subheader("💡 Nutrient Recommendations")
    for i, value in enumerate(features):
        key = feature_names[i]
        threshold = low_thresholds.get(key, 0)
        if value < threshold:
            st.markdown(f"- **{key} is Low** (Value: {value} < {threshold}): {recommendations.get(key, 'No suggestion')}")
