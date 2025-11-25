import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Step 1: Load Data, Model, AND Scaler ---
print("1. Loading data, model, and scaler...")
df = pd.read_csv('genetic_disease_dataset.csv')
X = df.drop('Disease', axis=1)
feature_names = X.columns.tolist() # Save names for plotting

# Load the saved artifacts
model = joblib.load('genetic_disease_model.pkl')
scaler = joblib.load('scaler.pkl') # We need this!

# Scale the data (Crucial: The model was trained on scaled data)
print("2. Scaling data to match model training...")
X_scaled = scaler.transform(X)
# Convert back to DataFrame just for convenience in SHAP (optional but good practice)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Extract the "Random Forest" doctor
rf_model = model.named_estimators_['rf']

# --- REPLACEMENT CODE FOR STEP 2 (Global Importance - MANUAL MARGINS) ---
print("3. Calculating Global Feature Importance...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled_df)

# Ensure shap_values is a list for multi-class plotting (Fixes Dimension issues)
shap_values_plot = shap_values
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values_plot = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# FIX 1: Use a much wider figure (16 inches) to give text room to breathe
plt.figure(figsize=(16, 10))

class_names = ['Thalassemia', 'Hemophilia', 'Breast Cancer', 'Sickle Cell', 'Cystic Fibrosis']

# FIX 2: Use modern colormap retrieval (Fixes the Deprecation Warning)
import matplotlib
cmap = matplotlib.colormaps['tab10']

# Draw the SHAP plot
shap.summary_plot(shap_values_plot, X_scaled_df, show=False, color=cmap)

# Create the custom legend
import matplotlib.patches as mpatches
legend_patches = []
for i, name in enumerate(class_names):
    patch = mpatches.Patch(color=cmap.colors[i], label=name)
    legend_patches.append(patch)

# FIX 3: Place Legend comfortably below the chart
# We use figure coordinates to center the legend on the IMAGE, not the AXES.
# This prevents the legend from being pushed off-screen due to the large left margin.
plt.legend(
    handles=legend_patches, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.02), # Bottom center of the FIGURE
    bbox_transform=plt.gcf().transFigure,
    ncol=5, 
    title="Disease Classes",
    fontsize=12
)

plt.title("Global Feature Importance by Disease", fontsize=18, pad=20)

# --- CRITICAL FIX: MANUAL MARGINS ---
# We disable 'tight_layout' and set the spacing manually.
# left=0.4:  Reserves 40% of the image width for the Feature Names (Left side)
# bottom=0.15: Reserves 15% of height for the Legend (Bottom)
# We increased bottom to 0.15 to make room for the legend.
plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.15)

# Save
plt.savefig('5_global_feature_importance_FIXED.png') # removed bbox_inches='tight' to respect our manual margins
plt.close()
print("✅ FINAL FIX: Global importance saved with manual margins.")

# --- Step 3: Local Explanation (Patient Case Study) ---
patient_index = 5
print(f"\n4. Analyzing Patient #{patient_index}...")

# Get the SCALED data for the patient (what the model sees)
patient_data_scaled = X_scaled[patient_index] 
# Reshape for prediction (1 sample, n features)
patient_data_reshaped = patient_data_scaled.reshape(1, -1)

# Get the RAW data for the patient (for us to read in the print statement)
patient_data_raw = X.iloc[patient_index]

# 1. Ask the model for a prediction
predicted_class_index = int(model.predict(patient_data_reshaped)[0])
predicted_disease_name = class_names[predicted_class_index]
actual_disease_index = int(df.iloc[patient_index]['Disease'])
actual_disease_name = class_names[actual_disease_index]

print(f"   Patient's True Diagnosis: {actual_disease_name} (Class {actual_disease_index})")
print(f"   Model's Prediction:       {predicted_disease_name} (Class {predicted_class_index})")
print(f"   Patient's Sweat Chloride: {patient_data_raw['Sweat_Chloride']}")
print(f"   Patient's Hemoglobin:     {patient_data_raw['Hemoglobin']}")

# Prepare data for plot
# Logic: We explain the class the model PREDICTED (to see why it made that choice)
class_index = predicted_class_index 

# Fix: shap_values is (n_samples, n_features, n_classes)
# We want the SHAP values for the specific patient and specific class
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_val_patient = shap_values[patient_index, :, class_index]
else:
    # Fallback if it is a list (old SHAP version behavior)
    shap_val_patient = shap_values[class_index][patient_index]

expected_val = explainer.expected_value[class_index]

plt.figure(figsize=(20, 3))
# We pass the RAW values (patient_data_raw) for the display labels,
# but the SHAP values (shap_val_patient) still determine the bar sizes.
# We round the raw values to 2 decimal places for cleaner reading.
shap.force_plot(
    expected_val, 
    shap_val_patient, 
    np.round(patient_data_raw.values, 2), # Show readable numbers (e.g., 29.29 instead of -1.88)
    feature_names=feature_names,
    matplotlib=True, 
    show=False,
    text_rotation=45
)
plt.savefig('6_patient_diagnosis_explanation.png', bbox_inches='tight')
plt.close()
print("✅ Patient diagnosis explanation saved as '6_patient_diagnosis_explanation.png'")