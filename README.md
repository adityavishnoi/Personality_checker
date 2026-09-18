# 🧠 Personality Prediction using Logistic Regression

A Machine Learning project that uses **Logistic Regression** to predict a person's personality type based on behavioral, social, and personal preference features.

The trained model was integrated into an interactive **Streamlit web application**, allowing users to answer questions about their personality and receive a predicted personality type.

## 📌 Project Overview

The objective of this project is to explore whether personality-related characteristics can be used to classify individuals into different personality types.

The project follows a complete Machine Learning workflow, from data preprocessing and feature selection to model training, evaluation, model serialization, and deployment.

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Pickle Files
   ↓
Streamlit Application
```

## 📊 Features Used

The model uses behavioral and personality-related attributes including:

* `social_energy`
* `alone_time_preference`
* `talkativeness`
* `deep_reflection`
* `group_comfort`
* `party_liking`
* `listening_skill`
* `empathy`
* `organization`
* `leadership`
* `risk_taking`
* `public_speaking_comfort`
* `curiosity`
* `routine_preference`
* `excitement_seeking`
* `friendliness`
* `planning`
* `spontaneity`
* `adventurousness`
* `reading_habit`
* `sports_interest`
* `online_social_usage`
* `travel_desire`
* `gadget_usage`
* `work_style_collaborative`
* `decision_speed`

The target variable is:

```text
personality_type
```

The target was encoded using **LabelEncoder** before training.

## 🤖 Model

**Algorithm:** Logistic Regression

Logistic Regression was used as a classification algorithm to learn the relationship between the behavioral features and personality categories.

The input features were scaled before being passed to the model.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression
* StandardScaler
* LabelEncoder
* Streamlit
* Pickle

## 📁 Project Structure

```text
personality-prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── encoder.pkl
├── requirements.txt
└── README.md
```

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can rate different personality traits.

The application provides:

* Interactive sliders
* Categorized personality questions
* Modern user interface
* Machine Learning prediction
* Prediction probabilities
* Automatic conversion of encoded predictions back to personality labels

## 🚀 Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd personality-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📦 Saved Model Files

The trained components are stored using Python's `pickle` module:

```text
model.pkl
scaler.pkl
encoder.pkl
```

They represent:

* `model.pkl` → trained Logistic Regression model
* `scaler.pkl` → feature scaling transformation
* `encoder.pkl` → LabelEncoder used for the personality target

During prediction, the application follows:

```text
User Input
    ↓
Scaling
    ↓
Logistic Regression
    ↓
Encoded Personality Class
    ↓
LabelEncoder.inverse_transform()
    ↓
Personality Type
```

## 📈 Model Performance

The model achieved approximately **99.75% accuracy** on the evaluated dataset.

Because such a high accuracy can sometimes indicate data leakage, dataset-specific patterns, or an especially easy classification problem, further validation should be performed before interpreting this performance as general real-world accuracy.

## 🎯 Key Learning Outcomes

This project helped me practice:

* Data preprocessing
* Feature selection
* Label encoding
* Feature scaling
* Logistic Regression
* Classification metrics
* Model serialization
* Streamlit application development
* Machine Learning deployment

## ⚠️ Disclaimer

This application is intended for **educational and demonstration purposes**. Personality predictions should not be treated as psychological assessments or professional evaluations.

## 👨‍💻 Author

**Aditya Vishnoi**

Built with Python, Scikit-learn and Streamlit. 🧠

Live Link- https://personalitychecker-tyhabspveptbbljkvatf6k.streamlit.app/
