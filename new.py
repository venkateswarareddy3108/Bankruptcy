import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load('xgb_best_model_main.pkl')

# Function to make predictions
def predict(client_data):
    prediction = model.predict(client_data)
    return prediction

# Streamlit app
st.title("Term Deposit Subscription Prediction")

st.write("""
This app predicts whether a client will subscribe to a term deposit based on the provided features.
""")

# Input fields for the selected 9 features

# 1. Numerical Inputs
duration = st.number_input('Duration of Last Contact (seconds)', min_value=0, max_value=5000, value=100, key='duration')
euribor3m = st.number_input('Euribor 3 Month Rate', min_value=0.0, max_value=10.0, value=1.0, key='euribor3m')
nr_employed = st.number_input('Number of Employees', min_value=0.0, max_value=10000.0, value=5000.0, key='nr_employed')
emp_var_rate = st.number_input('Employment Variation Rate', min_value=-5.0, max_value=5.0, value=1.0, key='emp_var_rate')
cons_conf_idx = st.number_input('Consumer Confidence Index', min_value=-50.0, max_value=50.0, value=-30.0, key='cons_conf_idx')

# Duration-based features
duration_squared = duration ** 2
duration_nr_employed = duration * nr_employed
duration_campaign = st.number_input('Duration Campaign (combination)', min_value=0, max_value=5000, value=100, key='duration_campaign')

# Collect all features into an array for prediction
client_data = np.array([[euribor3m, nr_employed,
                         duration_squared, duration_nr_employed,
                         emp_var_rate, cons_conf_idx, duration_campaign]])

# Make prediction
if st.button('Predict'):
    result = predict(client_data)
    st.write('Prediction:', 'Customer will Subscribe' if result == 1 else 'Customer will Not Subscribe')
