import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/alzheimers_disease_data.csv")

# Drop unwanted columns
df.drop(["PatientID", "DoctorInCharge"], axis=1, inplace=True)

# ----------------------------
# Select Only 9 Features (Same as Streamlit)
# ----------------------------
selected_features = [
    "Age",
    "Gender",
    "BMI",
    "Diabetes",
    "Depression",
    "PhysicalActivity",
    "SleepQuality",
    "MMSE",
    "MemoryComplaints"
]

X = df[selected_features]
y = df["Diagnosis"]

print("Training Features:", X.shape)

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ----------------------------
# Accuracy
# ----------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("✅ Model Accuracy:", acc)

# ----------------------------
# Save Model + Scaler
# ----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and Scaler Saved Successfully!")
