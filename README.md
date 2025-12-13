# 🩺 Multi-Disease Prediction System

> **A robust Machine Learning application to predict the risk of Diabetes, Heart Disease, and Breast Cancer with high accuracy.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

## 🌟 Overview

The **Multi-Disease Prediction System** is an AI-powered comprehensive health analysis tool. Built with **Streamlit** and **Scikit-Learn**, it leverages **Random Forest** algorithms to analyze patient medical data and predict the likelihood of three critical diseases.

The application features a user-friendly interface with strict medical range validation to ensure accurate data entry and reliable predictions.

---

## ✨ Features

*   **🏥 Multi-Disease Support**: Predict risks for three major conditions:
    *   **Diabetes**: Based on Glucose, BMI, Insulin, etc.
    *   **Heart Disease**: Analyzes Chest Pain, Cholesterol, ECG results, etc.
    *   **Breast Cancer**: Uses tumor metrics like Radius, Texture, and Smoothness.
*   **🛡️ Robust Validation**: Intelligent input forms enforce medically accepted ranges to prevent erroneous data.
*   **⚡ Real-Time Predictions**: Instant analysis using pre-trained Machine Learning models.
*   **📊 Confidence Scores**: Provides probability estimates alongside the prediction result (e.g., "Positive (85% Confidence)").
*   **📱 Responsive UI**: Clean, modern interface powered by Streamlit.

---

## �️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest Classifier)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Model Persistence**: [Joblib](https://joblib.readthedocs.io/)
*   **Data Source**: [KaggleHub](https://github.com/Kaggle/kagglehub)

---

## � Getting Started

Follow these steps to set up the project locally.

### 1. Prerequisites

*   Python 3.8 or higher.
*   pip (Python package installer).

### 2. Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/yourusername/disease-prediction-app.git
cd disease-prediction-app
pip install -r requirements.txt
```

### 3. Model Setup (Important)

Due to file size limits, the trained models are **not** included in the repo. You must generate them:

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Copy the code from `train_all_models.py` into a notebook.
3.  Run the script to train the models and generate **6 files**:
    *   `diabetes_model.pkl` & `diabetes_scaler.pkl`
    *   `heart_model.pkl` & `heart_scaler.pkl`
    *   `breast_cancer_model.pkl` & `breast_cancer_scaler.pkl`
4.  Download these files and place them in the root directory of this project.

### 4. Running the App

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```bash
disease-prediction-app/
├── 📂 .streamlit/          # Streamlit configuration
├── 📄 app.py               # Main application entry point
├── 📄 train_all_models.py  # Script to train models (run on Colab)
├── 📄 requirements.txt     # Python dependencies
├── 📄 DEPLOYMENT_GUIDE.md  # Instructions for cloud deployment
├── 📄 README.md            # Project documentation
└── 📦 *.pkl                # Model and Scaler files (generated)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
