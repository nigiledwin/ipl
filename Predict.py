import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved pipeline models
pipeline_models = {}
for model_name in ['Linear Regression', 'Random Forest', 'Ridge Regression', 'SVR', 'XGBoost', 'Gradient Boosting', 'Voting Regressor', 'Stacking Regressor', 'Blending Regressor']:
    filename = f'{model_name}.pkl'
    pipeline_model = joblib.load(filename)
    pipeline_models[model_name] = pipeline_model


# Streamlit UI
st.title('IPL Score Prediction App')

# User input section
st.sidebar.header('User Input')
batting_team = st.sidebar.selectbox('Select Batting Team', ['SRH', 'CSK', 'MI', 'RCB', 'KKR', 'RR', 'PBKS', 'DC'])
bowling_team = st.sidebar.selectbox('Select Bowling Team', ['SRH', 'CSK', 'MI', 'RCB', 'KKR', 'RR', 'PBKS', 'DC'])
is_bat_home_team = st.sidebar.selectbox('Is Batting Team Home Team?', ['Yes', 'No'])
over = st.sidebar.selectbox('Over', np.arange(1.0, 20.1, 1.0))
ball = st.sidebar.selectbox('Ball', np.arange(1, 7, 1))
total_runs = st.sidebar.selectbox('Total Runs', np.arange(0, 301, 10))
wkts = st.sidebar.selectbox('Wickets', np.arange(0, 11, 1))

# Submit button
submit_button = st.sidebar.button('Submit')

if submit_button:
    # Prepare user input data
    user_input = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'Is_bat_home_team': [is_bat_home_team],
        'over': [over],
        'ball': [ball],
        'total_runs': [total_runs],
        'Wkts': [wkts]
    })

    # Perform predictions using each model
    predictions = {}
    for model_name, pipeline_model in pipeline_models.items():
        y_pred = pipeline_model.predict(user_input)
        predictions[model_name] = np.round(y_pred)

    # Store predictions in a DataFrame
    predictions_df = pd.DataFrame(predictions).T
    predictions_df.columns = ['Prediction']

    # Display predictions DataFrame
    st.subheader('Predictions')
    st.write(predictions_df)

    # Calculate mean prediction
    mean_prediction = predictions_df['Prediction'].mean()

    # Display mean prediction
    st.subheader('Mean Prediction')
    st.write(mean_prediction)
