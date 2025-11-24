# ⛏️ Drilling Rate Predictor
Web app to predict drilling penetration rate using ML.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drilling-rate-predictorgit.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

## 📌 Overview
The **Drilling Rate Predictor** is an intelligent web application designed to estimate the **Rate of Penetration (ROP)** in rotary percussive drilling. By leveraging Machine Learning (XGBoost), this tool analyzes critical geological and machine parameters to provide accurate drilling speed predictions.

This project was developed as part of the **Drilling Technology Assignment** to identify critical parameters influencing drilling efficiency.

🔗 **Live Demo:** [Click Here to Access the App](https://drilling-rate-predictorgit.streamlit.app)

---

## 🚀 Key Features
* **🤖 AI-Powered Prediction:** Uses an **XGBoost Regressor** model trained on 15 distinct parameters to handle complex, non-linear relationships in rock mechanics.
* **🎨 Interactive UI:** Built with **Streamlit**, featuring a dark neon theme, 3D calculator display, and responsive design for a professional user experience.
* **⚡ Smart Defaults:** Includes auto-fill functionality for machine parameters to simplify testing.
* **📜 History Log:** Automatically saves prediction results (Time, ROP, UCS, Power) for session reference.
* **📄 Research Integration:** Contains a dedicated section referencing standard drilling research.

---

## ⚙️ How It Works
The model predicts the drilling rate based on the following input categories:

### 1. Machine Parameters
* Bit Diameter (mm)
* Rotary Drill Power (kW)
* Blow Frequency (bpm)
* Pulldown & Blow Pressure
* Rotational Pressure

### 2. Rock Properties
* Uniaxial Compressive Strength (UCS)
* Tensile Strength & Impact Strength
* Schmidt Rebound Number
* Elastic Modulus & Density
* P-Wave Velocity & Quartz Content

The system processes these inputs through the trained model (`drilling_prediction_model.pkl`) to output the estimated penetration rate in **m/min**.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **ML Engine:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

---

## 📦 Installation & Usage

To run this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Shivansh-Satyam/Drilling-Rate-Predictor.git](https://github.com/Shivansh-Satyam/Drilling-Rate-Predictor.git)
   cd Drilling-Rate-Predictor

## 📚 References

This project and its predictive methodology are supported by the following research:

> **"Rotary and Percussive Drilling Prediction Using Regression Analysis"**
> *S. Kahraman, International Journal of Rock Mechanics and Mining Sciences, 1999*

This study validates the use of regression-based models and key parameters (like UCS and operational settings) for predicting penetration rates.

---

## 👤 Author

**Shivansh Satyam** Roll No: 21 (BT24MIN021)

Department of Mining Engineering

---
*Created for Academic Submission | 2025*
