# eda_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load the Data ---
print("Loading dataset...")
df = pd.read_csv('genetic_disease_dataset.csv')

# --- Step 2: Decode the "Disease" Column ---
# Mapping 0-4 to actual names so the saved images are readable
disease_map = {
    0: 'Thalassemia',
    1: 'Hemophilia',
    2: 'Breast Cancer',
    3: 'Sickle Cell Anemia',
    4: 'Cystic Fibrosis'
}
df['Disease_Name'] = df['Disease'].map(disease_map)

# --- Step 3: Check for Class Imbalance (Saved as Image) ---
print("Generating Disease Distribution plot...")
plt.figure(figsize=(10, 6))
sns.countplot(x='Disease_Name', data=df, palette='viridis')
plt.title('Distribution of Patients per Disease')
plt.xticks(rotation=45)
plt.tight_layout()

# SAVE the plot instead of showing it
plt.savefig('1_disease_distribution.png') 
plt.close() # Close memory to prevent overlap


# --- Step 4: The Correlation Matrix (Saved as Image) ---
print("Generating Correlation Heatmap...")
plt.figure(figsize=(12, 10))
numeric_df = df.drop('Disease_Name', axis=1) 
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()

# SAVE the plot
plt.savefig('2_correlation_heatmap.png')
plt.close()


# --- Step 5: The "Smoking Gun" Evidence (Saved as Image) ---
print("Generating Biomarker Boxplots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cystic Fibrosis Check: Sweat Chloride
sns.boxplot(x='Disease_Name', y='Sweat_Chloride', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Sweat Chloride Levels (High in CF?)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Sickle Cell Check: Sickled RBC %
sns.boxplot(x='Disease_Name', y='Sickled_RBC_Percent', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Sickled RBC % (High in Sickle Cell?)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Breast Cancer Check: BRCA1 Expression
sns.boxplot(x='Disease_Name', y='BRCA1_Expression', data=df, ax=axes[1, 0])
axes[1, 0].set_title('BRCA1 Gene Expression (High in Cancer?)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Thalassemia Check: Fetal Hemoglobin
sns.boxplot(x='Disease_Name', y='Fetal_Hemoglobin', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Fetal Hemoglobin (High in Thalassemia?)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# SAVE the plot
plt.savefig('3_biomarker_analysis.png')
plt.close()

print("\nSUCCESS: All plots saved to your folder!")