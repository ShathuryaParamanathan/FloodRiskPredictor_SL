# 🌊 SL Flood Risk Predictor

## 📋 Project Overview
The **SL Flood Risk Predictor** is a production-ready machine learning system designed to predict and explain flood risk across Sri Lanka. By leveraging **XGBoost** and **SHAP**, the application provides high-precision risk scores (0-100) and human-friendly explanations for *why* specific locations are vulnerable.

This tool is designed for urban planners, disaster management units, and citizens to understand local flood dynamics based on real-time environmental and geographical factors.

---

## 🏗️ Project Structure
```text
FloodRiskPredictor_SL/
├── data/               # Raw and cleaned datasets
├── models/             # Trained XGBoost models and Scalers
├── scripts/            # Preprocessing and utility scripts
├── report_assets/      # Visualizations for documentation
├── streamlit_app.py    # Interactive Web Dashboard
├── train_model.py      # ML Training Pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── .dockerignore       # Docker build exclusions
```

---

## 📊 Dataset: Sri Lanka Flood Risk & Inundation
We used the **Sri Lanka Flood Risk & Inundation Dataset** from Kaggle, containing **25,000+ records** covering all districts in Sri Lanka.

- **Geographic:** Latitude, Longitude, Elevation, Distance to River.
- **Environmental:** 7-day Rainfall, Monthly Rainfall, NDVI (Vegetation), NDWI (Water).
- **Infrastructural:** Population Density, Built-up %, Infrastructure Score.
- **Target Variable:** `flood_risk_score` (Continuous value 0.0 – 100.0).

---

## 🤖 Machine Learning Model
- **Algorithm:** XGBoost Regressor (eXtreme Gradient Boosting).
- **Performance:** 
    - **MAE:** 0.799
    - **R² Score:** 0.9889
- **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** to provide transparency for every prediction.

---

## 🚀 How to Run (Local Setup)

### 1. Prerequisites
- Python 3.12+ 
- Virtual environment (recommended)

### 2. Installation
```bash
# Clone the repository
cd FloodRiskPredictor_SL

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run streamlit_app.py
```
*The app will be available at [http://localhost:8501](http://localhost:8501)*

---

## 🐳 Running with Docker
Containerization ensures the app runs consistently across any system with all libraries (XGBoost/SHAP) pre-configured.

### 1. Build the Image
```powershell
docker build -t flood-risk-predictor .
```

### 2. Run the Container
```powershell
docker run -p 8501:8501 flood-risk-predictor
```

### 3. Access the App
Go to: **[http://localhost:8501](http://localhost:8501)**

---

## 🕵️ Explainability Features
Unlike traditional "black box" models, this application tells you the **why**:
- **Human-friendly Summary:** Translates complex statistics into sentences (e.g., *"Your risk is high primarily due to low elevation and high 7d rainfall"*).
- **Technical Impact Graph:** Shows exactly which features (rainfall, soil type, etc.) pushed the risk score up or down.

---

## 🛠️ Developed for
Machine Learning Module | University of Moratuwa | Information Technology
