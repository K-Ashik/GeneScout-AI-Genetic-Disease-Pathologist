# 🧬 GeneScout: AI-Powered Genetic Disease Pathologist

## 🚀 Project Overview
GeneScout is an interpretable Machine Learning diagnostic tool designed to predict 5 genetic diseases (Cystic Fibrosis, Sickle Cell, etc.) based on patient biomarkers. Unlike "Black Box" models, GeneScout prioritizes clinical explainability using SHAP values.

## 🛠️ Tech Stack
* **Model:** Voting Ensemble (Random Forest + SVM + Logistic Regression)
* **Explainability:** SHAP (Shapley Additive exPlanations) for global and local feature importance.
* **Deployment:** Streamlit Web App for real-time inference.
* **Data Analysis:** Pandas, Seaborn, Matplotlib.

## 📊 Key Findings
1.  **Sweat Chloride** was identified as the primary biomarker for Cystic Fibrosis (SHAP value > 0.8).
2.  **Hemoglobin & Fetal Hemoglobin** levels successfully differentiated Thalassemia from anemia.
3.  **Accuracy:** The ensemble model achieved **93.5% Accuracy** on the test set.

## 🖼️ App Screenshot
*(Put your screenshot here!)*
![app_view](https://github.com/user-attachments/assets/f18c955b-e5e0-47cc-a8e6-79aebc764d87)


## 📂 Project Structure
* `app.py`: The Streamlit dashboard.
* `train_model.py`: Training script for the Voting Classifier.
* `explain_model.py`: SHAP analysis and plot generation.
* `eda_analysis.py`: Initial data exploration.
