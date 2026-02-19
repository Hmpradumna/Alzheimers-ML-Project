# Alzheimers-ML-Project

A Machine Learning based web application that predicts the risk of Alzheimer’s Disease using patient demographic, lifestyle, and cognitive assessment data.

This project helps in early detection by analyzing important medical and health indicators.

---

## 📌 Project Objective

The goal of this project is to build a classification model that can identify patients at high risk of Alzheimer’s Disease based on:

- Demographic details (Age, Gender, BMI)
- Lifestyle factors (Physical Activity, Sleep Quality)
- Medical history (Diabetes, Depression)
- Cognitive scores (MMSE, Memory Complaints)

---

## 📂 Project Structure

Alzheimers-ML-Project/
│
├── data/
│ └── alzheimers_disease_data.csv
│
├── models/
│ ├── best_model.pkl
│ └── scaler.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md


---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit (Web Deployment)
- Joblib (Model Saving)

---

## 🚀 How to Run the Project Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
