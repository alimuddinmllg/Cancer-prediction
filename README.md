# Breast Cancer Predictor App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cancer-prediction-kzyjudzvgmgbxf6edjrjhg.streamlit.app/)

A machine learning-powered web application that helps predict breast cancer (benign/malignant) based on cell nuclei measurements from cytology labs.

![App Screenshot](https://via.placeholder.com/800x400.png?text=Breast+Cancer+Prediction+App+Screenshot)

## Features

- Interactive sidebar with measurement sliders
- Real-time radar chart visualization of cell nuclei features
- Machine learning predictions with probability scores
- Responsive and user-friendly interface
- Three measurement categories: Mean, Standard Error, and Worst
- Clear visualization of feature comparisons

## Installation

To run locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:

bash
Copy
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
streamlit run app.py
Usage
Adjust the sliders in the sidebar to input cell nuclei measurements

View the interactive radar chart showing three measurement types

See real-time predictions in the right column

Interpret results:

Benign: Non-cancerous diagnosis

Malignant: Cancerous diagnosis

Probability scores show prediction confidence

Note: This app should be used as a decision support tool, not a replacement for professional medical diagnosis.

Dataset
Uses the Wisconsin Breast Cancer Dataset with preprocessing:

Removed unnecessary columns (id, Unnamed: 32)

Mapped diagnosis to binary values (M=1, B=0)

Features normalized for model input

Technologies Used
Python 3.9+

Streamlit (Web Framework)

Plotly (Interactive Visualizations)

Scikit-Learn (Machine Learning)

Pandas (Data Processing)

NumPy (Numerical Operations)

Model Details
Logistic Regression classifier

Trained on standardized features

Persistent model using pickle serialization

Input features scaled using MinMaxScaler

Deployment
The app is deployed on Streamlit Cloud and can be accessed at:
https://cancer-prediction-kzyjudzvgmgbxf6edjrjhg.streamlit.app/

For Medical Professionals
This tool aims to:

Provide quick preliminary analysis

Visualize feature relationships

Support clinical decision-making

Reduce diagnostic time

Important: Always verify app predictions with clinical assessments and laboratory tests.
