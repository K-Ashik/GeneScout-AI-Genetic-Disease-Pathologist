# train_model.py

import pandas as pd
import joblib  # Used to save the model for the app later
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Load Data ---
print("1. Loading data...")
df = pd.read_csv('genetic_disease_dataset.csv')

# Separate Features (X) and Target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# --- Step 2: Split Data ---
# We keep 20% of data hidden to test the doctors later.
# stratify=y ensures we have an equal number of sick people in both sets.
print("2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 3: Scaling (Crucial) ---
# SVM and Logistic Regression fail if numbers are too big/small.
# We scale everything to have a mean of 0 (Standard Deviation).
print("3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Define the "Doctors" ---
print("4. Initializing the Board of Doctors...")

# Doctor 1: Logistic Regression (The Statistician)
# Good for finding simple linear relationships.
clf1 = LogisticRegression(random_state=1)

# Doctor 2: Random Forest (The Specialist)
# Good for complex rules (if Age > 50 and Gene = Mutated...)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

# Doctor 3: SVM (The Mathematician)
# Draws complex geometric boundaries between diseases.
# probability=True is required so it can "vote" with confidence scores.
clf3 = SVC(kernel='linear', probability=True, random_state=1)

# --- Step 5: The Voting Classifier ---
# voting='soft' means we average the probabilities (e.g., 90% + 80% + 70%)
# instead of just counting "Yes/No" votes. It's more accurate.
ensemble_model = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], 
    voting='soft'
)

# --- Step 6: Train ---
print("5. Training models (this might take a second)...")
ensemble_model.fit(X_train_scaled, y_train)

# --- Step 7: Evaluate ---
print("6. Evaluating performance...")
y_pred = ensemble_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --- Step 8: Save the Brain ---
# We save the Model AND the Scaler. We need both for the App.
joblib.dump(ensemble_model, 'genetic_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and Scaler saved to disk!")

# Optional: Save Confusion Matrix Image
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Board of Doctors')
plt.ylabel('Actual Disease')
plt.xlabel('Predicted Disease')
plt.savefig('4_confusion_matrix.png')
print("✅ Confusion Matrix saved as image.")