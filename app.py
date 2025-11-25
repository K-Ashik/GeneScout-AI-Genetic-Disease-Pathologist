import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the "Brain" (Model & Scaler) ---
@st.cache_resource
def load_artifacts():
    # We load the model and the scaler we saved during training
    model = joblib.load('genetic_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Error: Model files not found. Please run 'train_model.py' first!")
    st.stop()

# --- 2. App Title & Description ---
st.set_page_config(page_title="GeneScout AI", layout="wide", page_icon="🧬")

st.title("🧬 GeneScout: AI Genetic Disease Pathologist")
st.markdown("""
### Precision Diagnostics from Biomarkers
This AI system assists doctors in predicting **5 Genetic Disorders** by analyzing biochemical markers.
Adjust the patient vitals on the sidebar to see the AI's diagnosis in real-time.
""")

# --- 3. Sidebar: Patient Vitals (Inputs) ---
st.sidebar.header("Patient Vitals")
st.sidebar.markdown("---")

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (Years)", 1, 80, 30)
gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
gender_val = 0 if gender == "Male" else 1
family_history = st.sidebar.radio("Family History?", ["No", "Yes"], horizontal=True)
history_val = 1 if family_history == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.subheader("🩸 Blood Panel")
hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 4.0, 20.0, 12.0, help="Normal range: 12-16 g/dL")
fetal_hemoglobin = st.sidebar.slider("Fetal Hemoglobin (%)", 0.0, 20.0, 1.0, help="High levels indicate Thalassemia")
rdw = st.sidebar.slider("RDW-CV (%)", 10.0, 30.0, 14.0, help="Red Cell Distribution Width")
ferritin = st.sidebar.slider("Serum Ferritin (ng/mL)", 10.0, 300.0, 100.0)
sickled_rbc = st.sidebar.slider("Sickled RBC (%)", 0.0, 40.0, 0.0, help="Percentage of Sickle Cells")

st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Genetic & Special Markers")
brca1 = st.sidebar.slider("BRCA1 Expression", 0.0, 2.0, 0.2, help="High levels linked to Breast Cancer")
p53 = st.sidebar.radio("p53 Mutation?", ["Normal", "Mutated"], horizontal=True)
p53_val = 1 if p53 == "Mutated" else 0
sweat_chloride = st.sidebar.slider("Sweat Chloride (mmol/L)", 0.0, 100.0, 20.0, help=">60 is diagnostic for Cystic Fibrosis")
il6 = st.sidebar.slider("IL-6 Level (pg/mL)", 0.0, 30.0, 5.0)

# --- 4. Prepare the Data for the Model ---
# The order MUST match the training data exactly!
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_val],
    'Family_History': [history_val],
    'Hemoglobin': [hemoglobin],
    'Fetal_Hemoglobin': [fetal_hemoglobin],
    'RDW_CV': [rdw],
    'Serum_Ferritin': [ferritin],
    'BRCA1_Expression': [brca1],
    'p53_Mutation': [p53_val],
    'Sweat_Chloride': [sweat_chloride],
    'Sickled_RBC_Percent': [sickled_rbc],
    'IL6_Level': [il6]
})

# --- 5. Make the Prediction ---
# Scale the input using the loaded scaler (CRITICAL STEP)
input_scaled = scaler.transform(input_data)

# Predict Class and Probability
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

# Map class ID to Name
disease_map = {
    0: 'Thalassemia',
    1: 'Hemophilia',
    2: 'Breast Cancer',
    3: 'Sickle Cell Anemia',
    4: 'Cystic Fibrosis'
}
diagnosis = disease_map[prediction]
confidence = probabilities[prediction] * 100

# --- 6. Display Results ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 🔍 Diagnostic Result")
    
    # Dynamic Color based on Confidence
    if confidence > 85:
        st.success(f"### Diagnosis: {diagnosis}")
    elif confidence > 60:
        st.warning(f"### Diagnosis: {diagnosis} (Moderate Confidence)")
    else:
        st.error(f"### Diagnosis: {diagnosis} (Low Confidence)")
    
    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
    
    # Clinical Recommendation Engine
    st.info("**Next Steps:**")
    if diagnosis == "Cystic Fibrosis":
        st.write("• Refer to Pulmonologist.")
        st.write("• Order confirmatory Sweat Test.")
    elif diagnosis == "Sickle Cell Anemia":
        st.write("• Refer to Hematologist.")
        st.write("• Pain management protocol.")
    elif diagnosis == "Thalassemia":
        st.write("• Complete Blood Count (CBC) review.")
        st.write("• Genetic counseling for globin genes.")
    elif diagnosis == "Breast Cancer":
        st.write("• Urgent Referral to Oncologist.")
        st.write("• Mammography and Biopsy.")
    elif diagnosis == "Hemophilia":
        st.write("• Clotting Factor test.")
        st.write("• Avoid blood thinners.")

with col2:
    st.markdown("### 📊 Probability Breakdown")
    # Create a clean DataFrame for the chart
    prob_df = pd.DataFrame({
        'Disease': list(disease_map.values()),
        'Probability': probabilities
    })
    
    # Plot using Seaborn
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(x='Probability', y='Disease', data=prob_df, palette='viridis', ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.set_xlabel("Probability Score")
    st.pyplot(fig)