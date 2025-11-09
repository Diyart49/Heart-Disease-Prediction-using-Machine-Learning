# CARDIOCARE - Heart-Disease-Prediction-using-Machine-Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


CARDIOCARE named heart disease prediction app 
Author- Diya Rastogi
Despite the continuous advancements in medical technology, heart disease persists as a leading cause of death worldwide. Many individuals remain unaware of their risk factors or lack access to timely screenings. Additionally, there is a shortage of personalized tools available to accurately assess individual risk, resulting in a gap in proactive heart health management.
To address this issue, CARDIOCARE offers a user-friendly heart disease prediction application. By employing Python for the backend, the app efficiently gathers and analyzes users' health parameters to generate personalized risk assessments. Utilizing machine learning techniques, the app's algorithm processes the data to predict the likelihood of heart disease for each user.

I've used a variety of Machine Learning algorithms, implemented in Python, to predict the presence of heart disease in a patient. This is a classification problem, with input features as a variety of parameters, and the target variable as a binary variable, predicting whether heart disease is present or not.

## 🧩 Features
- Interactive web interface for data input  
- ML model trained on a healthcare dataset  
- Real-time prediction results  
- End-to-end integration of model with Flask  
- Clean, modular, and reproducible code  

---

## ⚙️ Tech Stack
**Languages & Libraries:** Python, NumPy, Pandas, scikit-learn, Flask  
**Frontend:** HTML, CSS  
**Concepts Used:** Feature Scaling, Train-Test Split, Model Evaluation  

---

## 📈 Model Overview
- Trained multiple models (Logistic Regression, Random Forest, XGBoost)  
- Selected best-performing model based on **ROC-AUC** and **F1-Score**  
- Final model deployed in Flask web app  

---

## 🚀 How to Run Locally

```bash
# Clone this repository
git clone https://github.com/diyart49/Heart-Disease-Prediction-ML-App.git

# Navigate into the project directory
cd Heart-Disease-Prediction-ML-App

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py





