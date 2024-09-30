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

# Input fields for all 47 features

# 1. Numerical Inputs
age = st.number_input('Age', min_value=18, max_value=100, value=30, key='age')
duration = st.number_input('Duration of Last Contact (seconds)', min_value=0, max_value=5000, value=100, key='duration')
campaign = st.number_input('Number of Contacts During Campaign', min_value=1, max_value=50, value=1, key='campaign')
pdays = st.number_input('Days Since Previous Campaign Contact', min_value=-1, max_value=1000, value=-1, key='pdays')
previous = st.number_input('Number of Contacts Before Campaign', min_value=0, max_value=100, value=0, key='previous')
emp_var_rate = st.number_input('Employment Variation Rate', min_value=-5.0, max_value=5.0, value=1.0, key='emp_var_rate')
cons_price_idx = st.number_input('Consumer Price Index', min_value=90.0, max_value=100.0, value=93.0, key='cons_price_idx')
cons_conf_idx = st.number_input('Consumer Confidence Index', min_value=-50.0, max_value=50.0, value=-30.0, key='cons_conf_idx')
euribor3m = st.number_input('Euribor 3 Month Rate', min_value=0.0, max_value=10.0, value=1.0, key='euribor3m')
nr_employed = st.number_input('Number of Employees', min_value=0.0, max_value=10000.0, value=5000.0, key='nr_employed')

# 2. Categorical Inputs
job = st.selectbox('Job', ['admin', 'technician', 'blue-collar', 'management', 'retired', 'self-employed', 'entrepreneur', 'services', 'student', 'unemployed'], key='job')
housing = st.selectbox('Has Housing Loan?', ['yes', 'no'], key='housing')
loan = st.selectbox('Has Personal Loan?', ['yes', 'no'], key='loan')
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'], key='contact')
month = st.selectbox('Month of Last Contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], key='month')
day_of_week = st.selectbox('Day of Week of Last Contact', ['mon', 'tue', 'wed', 'thu', 'fri'], key='day_of_week')
poutcome = st.selectbox('Previous Outcome', ['failure', 'nonexistent', 'success'], key='poutcome')

# 3. Encoded Marital Status
marital_1 = st.selectbox('Marital Status: Single', ['yes', 'no'], key='marital_1')
marital_2 = st.selectbox('Marital Status: Married', ['yes', 'no'], key='marital_2')
marital_3 = st.selectbox('Marital Status: Divorced', ['yes', 'no'], key='marital_3')

# 4. Encoded Education Status
education_1 = st.selectbox('Education: Primary', ['yes', 'no'], key='education_1')
education_2 = st.selectbox('Education: Secondary', ['yes', 'no'], key='education_2')
education_3 = st.selectbox('Education: Tertiary', ['yes', 'no'], key='education_3')
education_4 = st.selectbox('Education_4', ['yes', 'no'], key='education_4')
education_5 = st.selectbox('Education_5', ['yes', 'no'], key='education_5')
education_6 = st.selectbox('Education_6', ['yes', 'no'], key='education_6')
education_7 = st.selectbox('Education_7', ['yes', 'no'], key='education_7')

# 5. Default Status
default_1 = st.selectbox('Has Credit in Default?', ['yes', 'no'], key='default_1')
default_2 = st.selectbox('Default_2', ['yes', 'no'], key='default_2')

# 6. Previous Contact Info
previous_1 = st.selectbox('Previous_1', ['yes', 'no'], key='previous_1')
previous_2 = st.selectbox('Previous_2', ['yes', 'no'], key='previous_2')
previous_3 = st.selectbox('Previous_3', ['yes', 'no'], key='previous_3')

# 7. Feature Engineering Inputs
duration_nr_employed = st.number_input('Duration * Nr Employed', value=0.0, key='duration_nr_employed')
campaign_pdays = st.number_input('Campaign * Pdays', value=0.0, key='campaign_pdays')
duration_1 = st.number_input('Duration.1', value=0.0, key='duration_1')
campaign_1 = st.number_input('Campaign.1', value=0.0, key='campaign_1')
pdays_1 = st.number_input('Pdays.1', value=0.0, key='pdays_1')

# 8. Interaction Terms
duration_squared = st.number_input('Duration Squared', value=0.0, key='duration_squared')
duration_campaign = st.number_input('Duration * Campaign', value=0.0, key='duration_campaign')
duration_pdays = st.number_input('Duration * Pdays', value=0.0, key='duration_pdays')
campaign_squared = st.number_input('Campaign Squared', value=0.0, key='campaign_squared')
campaign_pdays_interaction = st.number_input('Campaign * Pdays Interaction', value=0.0, key='campaign_pdays_interaction')
pdays_squared = st.number_input('Pdays Squared', value=0.0, key='pdays_squared')

# 9. Month and Day Features
month_sin = st.number_input('Month Sine', value=0.0, key='month_sin')
month_cos = st.number_input('Month Cosine', value=0.0, key='month_cos')
day_of_week_sin = st.number_input('Day of Week Sine', value=0.0, key='day_of_week_sin')
day_of_week_cos = st.number_input('Day of Week Cosine', value=0.0, key='day_of_week_cos')

# When the button is clicked, make a prediction
if st.button("Predict"):
    # Convert categorical 'yes'/'no' inputs to binary 1/0
    def convert_binary(value):
        return 1 if value == 'yes' else 0

    # Convert categorical inputs
    marital_1 = convert_binary(marital_1)
    marital_2 = convert_binary(marital_2)
    marital_3 = convert_binary(marital_3)
    education_1 = convert_binary(education_1)
    education_2 = convert_binary(education_2)
    education_3 = convert_binary(education_3)
    education_4 = convert_binary(education_4)
    education_5 = convert_binary(education_5)
    education_6 = convert_binary(education_6)
    education_7 = convert_binary(education_7)
    default_1 = convert_binary(default_1)
    default_2 = convert_binary(default_2)
    previous_1 = convert_binary(previous_1)
    previous_2 = convert_binary(previous_2)
    previous_3 = convert_binary(previous_3)
    housing = convert_binary(housing)
    loan = convert_binary(loan)

    # Encode other categorical variables using one-hot encoding or label encoding as per your training
    # For simplicity, let's assume they were label encoded
    job_mapping = {'admin': 0, 'technician': 1, 'blue-collar': 2, 'management': 3, 'retired': 4, 'self-employed': 5, 'entrepreneur': 6, 'services': 7, 'student': 8, 'unemployed': 9}
    contact_mapping = {'cellular': 0, 'telephone': 1}
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_of_week_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
    poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2}

    job = job_mapping[job]
    contact = contact_mapping[contact]
    month = month_mapping[month]
    day_of_week = day_of_week_mapping[day_of_week]
    poutcome = poutcome_mapping[poutcome]

    # Prepare the input data array
    client_data = np.array([[age, duration, campaign, pdays, previous, emp_var_rate, cons_price_idx,
                             cons_conf_idx, euribor3m, nr_employed, job, housing, loan, contact, month,
                             day_of_week, poutcome, marital_1, marital_2, marital_3, education_1, education_2,
                             education_3, education_4, education_5, education_6, education_7, default_1,
                             default_2, previous_1, previous_2, previous_3, duration_nr_employed, campaign_pdays,
                             duration_1, campaign_1, pdays_1, duration_squared, duration_campaign, duration_pdays,
                             campaign_squared, campaign_pdays_interaction, pdays_squared, month_sin, month_cos,
                             day_of_week_sin, day_of_week_cos]])

    # Ensure the input shape matches the model's expected feature shape
    if client_data.shape[1] != 47:
        st.error(f"Feature shape mismatch: expected 47, got {client_data.shape[1]}")
    else:
        # Make prediction
        prediction = predict(client_data)

        # Display the result
        if prediction == 1:
            st.success("The client is likely to subscribe to the term deposit.")
        else:
            st.error("The client is unlikely to subscribe to the term deposit.")
