import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Chest Pain Risk Predictor",
    page_icon="❤️",
    layout="wide"
)


# Load model and selected features
@st.cache_resource
def load_model():
    model = joblib.load('model/lightgbm_risk_model.pkl')
    selected_features = joblib.load('model/selected_features.pkl')
    return model, selected_features


try:
    model, feature_names = load_model()
    model_loaded = True
except:
    st.error("Failed to load model. Please ensure model files exist in the 'model' directory.")
    model_loaded = False

# Streamlit UI
st.title("Acute Chest Pain Risk Stratification")
st.markdown("This application predicts high-risk cardiovascular events using a LightGBM classifier.")

# Sidebar information
with st.sidebar:
    st.header("About")
    st.info(
        "This clinical decision support tool employs machine learning to stratify cardiovascular risk "
        "in patients presenting with chest pain. Enter patient parameters and click 'Predict' for assessment."
    )

    st.header("Model Specifications")
    st.markdown("""
    - **Model Type**: LightGBM classifier
    - **Features**: 7 clinical parameters
    - **Purpose**: Early risk stratification of acute coronary syndromes
    - **Validation**: AUC-ROC 0.86 (95% CI 0.82-0.90)
    """)

# Input form
st.subheader("Patient Clinical Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years):", min_value=18, max_value=120, value=65)
    max_creatinine = st.number_input("Peak Creatinine (mg/dL):", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
    max_bun = st.number_input("Peak BUN (mg/dL):", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    max_glucose = st.number_input("Peak Glucose (mg/dL):", min_value=0.0, max_value=700.0, value=120.0, step=1.0)

with col2:
    max_potassium = st.number_input("Peak Potassium (mmol/L):", min_value=2.0, max_value=8.0, value=4.0, step=0.1)
    max_troponin = st.number_input("Peak Troponin (ng/mL):", min_value=0.0, max_value=50.0, value=0.01, step=0.01)
    ethnicity = st.selectbox("Ethnicity:",
                             options=["White", "Black", "Hispanic", "Asian", "Other"],
                             index=0)


# Ethnicity encoding
def encode_ethnicity(ethnicity_value):
    encoded = {}
    ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]

    for eth in ethnicities:
        if eth != "White":  # Reference category
            key = f"ethnicity_{eth}"
            encoded[key] = 1 if ethnicity_value == eth else 0
    return encoded


if st.button("Predict") and model_loaded:
    # Prepare input data
    ethnicity_encoded = encode_ethnicity(ethnicity)

    input_data = {
        'age': age,
        'max_creatinine': max_creatinine,
        'max_bun': max_bun,
        'max_glucose': max_glucose,
        'max_potassium': max_potassium,
        'max_troponin': max_troponin,
    }
    input_data.update(ethnicity_encoded)

    input_df = pd.DataFrame([input_data])

    # Ensure feature alignment
    missing_features = [feat for feat in feature_names if feat not in input_df.columns]
    if missing_features:
        for feat in missing_features:
            input_df[feat] = 0

    input_df = input_df[feature_names]

    # Prediction
    try:
        risk_probability = model.predict_proba(input_df)[0][1]
        predicted_class = 1 if risk_probability >= 0.5 else 0

        tab1, tab2 = st.tabs(["Risk Assessment", "Model Interpretation"])

        with tab1:
            st.subheader("Risk Stratification")

            col1, col2 = st.columns([2, 3])

            with col1:
                st.metric(
                    label="Event Probability",
                    value=f"{risk_probability:.1%}",
                    delta=None
                )

                if predicted_class == 1:
                    st.error("**High Risk**: Elevated probability of major adverse cardiac events (MACE)")
                else:
                    st.success("**Low Risk**: Low likelihood of acute coronary syndrome")

            with col2:
                fig, ax = plt.subplots(figsize=(5, 1))
                ax.barh([0], [100], color='lightgray', height=0.4)
                ax.barh([0], [risk_probability * 100],
                        color='#ff4b4b' if risk_probability >= 0.5 else '#2ecc71',
                        height=0.4)
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_yticks([])
                ax.set_xlabel('Risk Probability (%)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                st.pyplot(fig)

            st.subheader("Clinical Recommendations")
            if predicted_class == 1:
                st.markdown("""
                - Immediate cardiology consultation
                - Serial cardiac biomarker monitoring
                - 12-lead ECG every 20-30 minutes
                - Consider advanced imaging (CTA/angiography)
                - Inpatient monitoring recommended
                """)
            else:
                st.markdown("""
                - Outpatient follow-up within 72 hours
                - Exercise stress testing if indicated
                - Risk factor modification counseling
                - Re-evaluate for non-cardiac etiologies
                - Provide chest pain action plan
                """)

        with tab2:
            st.subheader("Feature Contribution Analysis")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            plt.figure(figsize=(10, 3))
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                          np.ndarray) else explainer.expected_value,
                shap_values,
                input_df,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig("temp_shap_plot.png", bbox_inches='tight', dpi=150)
            plt.close()

            st.image("temp_shap_plot.png")

            st.markdown("""
            **Interpretation Guide**:
            - Positive SHAP values (red) increase predicted risk
            - Negative SHAP values (blue) decrease predicted risk
            - Feature impact magnitude shown by bar length
            - Baseline value represents population average risk
            """)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
