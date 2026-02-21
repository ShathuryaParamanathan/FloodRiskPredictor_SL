import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="SL Flood Risk Predictor Dashboard", layout="wide")

# Load model and metadata
@st.cache_resource
def load_assets():
    models_dir = r'd:\Academics\L4S1\ML\Assignment\FloodRiskPredictor_SL\models'
    model = joblib.load(os.path.join(models_dir, 'xgboost_flood_model.pkl'))
    explainer = joblib.load(os.path.join(models_dir, 'shap_explainer.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    default_values = joblib.load(os.path.join(models_dir, 'default_values.pkl'))
    return model, explainer, feature_columns, default_values

try:
    model, explainer, feature_columns, default_values = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ✅ PART 3 — Prediction Function
def get_prediction_data(input_dict):
    # Create DataFrame from input
    input_df = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    input_df = input_df[feature_columns].astype(float)
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(100, prediction))
    
    # Compute SHAP values
    # For PermutationExplainer/Explainer(model.predict), we pass the DF
    shap_values = explainer(input_df)
    
    return prediction, shap_values, input_df

# ✅ PART 4 — PROFESSIONAL STREAMLIT FRONTEND
st.title("🌊 SL Flood Risk Predictor Dashboard")
st.markdown("#### Predict flood risk using environmental and infrastructure indicators.")
st.markdown("---")

# UI Layout: Input section
st.subheader("🛠️ Environmental & Infrastructure Parameters")
st.info("Inputs are pre-filled with dataset averages. Adjust sliders to match your scenario.")

# Create columns for grouping
col1, col2, col3 = st.columns(3)

user_inputs = default_values.copy()

with col1:
    st.markdown("### 🌍 Geographic Factors")
    user_inputs['elevation_m'] = st.number_input("Elevation (scaled)", value=float(default_values['elevation_m']), format="%.4f")
    user_inputs['distance_to_river_m'] = st.number_input("Distance to River (scaled)", value=float(default_values['distance_to_river_m']), format="%.4f")

with col2:
    st.markdown("### 🌧 Rainfall Conditions")
    user_inputs['rainfall_7d_mm'] = st.number_input("Recent Rainfall (7 days) (scaled)", value=float(default_values['rainfall_7d_mm']), format="%.4f")
    user_inputs['monthly_rainfall_mm'] = st.number_input("Monthly Rainfall (scaled)", value=float(default_values['monthly_rainfall_mm']), format="%.4f")
    user_inputs['drainage_index'] = st.slider("Drainage Index", 0.0, 1.0, float(default_values['drainage_index']))

with col3:
    st.markdown("### 🏙 Urban & Population")
    user_inputs['population_density_per_km2'] = st.number_input("Population Density (scaled)", value=float(default_values['population_density_per_km2']), format="%.4f")
    user_inputs['built_up_percent'] = st.slider("Built-up Percentage", 0.0, 100.0, float(default_values['built_up_percent']))

st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.markdown("### 🏥 Infrastructure Access")
    user_inputs['infrastructure_score'] = st.slider("Infrastructure Score", 0.0, 100.0, float(default_values['infrastructure_score']))
    user_inputs['nearest_hospital_km'] = st.number_input("Distance to Hospital (km/scaled)", value=float(default_values['nearest_hospital_km']), format="%.4f")
    user_inputs['nearest_evac_km'] = st.number_input("Distance to Evac Center (km/scaled)", value=float(default_values['nearest_evac_km']), format="%.4f")

with col5:
    st.markdown("### 🌱 Environmental Indicators (Advanced)")
    with st.expander("Advanced Satellite Data"):
        user_inputs['ndvi'] = st.slider("NDVI (Vegetation Index)", -1.0, 1.0, float(default_values['ndvi']))
        user_inputs['ndwi'] = st.slider("NDWI (Water Index)", -1.0, 1.0, float(default_values['ndwi']))

# Predict Button
if st.button("🚀 Predict Flood Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing risk factors..."):
        prediction, shap_values, input_df = get_prediction_data(user_inputs)
        
        # Result section
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("## Prediction Result")
            st.metric("Flood Risk Score", f"{prediction:.2f}")
            
            if prediction <= 30:
                st.success("Risk Category: **Low Risk** ✅")
            elif prediction <= 60:
                st.warning("Risk Category: **Medium Risk** ⚠️")
            else:
                st.error("Risk Category: **High Risk** 🚨")
            
            st.progress(prediction / 100.0)
            
        with res_col2:
            st.markdown("## 📊 Explanation (SHAP)")
            st.write("This plot shows which features contributed most to this specific prediction.")
            
            # Local Explanation Plot (Waterfall)
            fig, ax = plt.subplots(figsize=(10, 6))
            # shap_values[0] because we only have one row
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)

        # Feature Importance Section
        st.markdown("### 🔍 Factors Influencing This Prediction")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        # Summary plot for the single instance basically shows which features pushed it up/down
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig2)

st.sidebar.title("About")
st.sidebar.info("""
This dashboard uses an **XGBoost Regression Model** to predict the flood risk score for specific locations in Sri Lanka.
- **SHAP (SHapley Additive exPlanations)** is used to explain the contribution of each environmental factor.
- Red bars increase risk, Blue bars decrease risk.
""")
