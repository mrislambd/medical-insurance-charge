import streamlit as st
import pickle
import pandas as pd
import numpy as np


def prediction():
    st.title('Medical Insurance Charge Prediction ')
    st.write("### Input the customer information")

    gender = ("male", "female")
    smoke = ("yes", "no")
    child = [0, 1, 2, 3, 4, 5]
    regions = ("northeast", "northwest", "southeast", "southwest")

    # Input fields
    age = st.number_input(label="Age: minimum 18 years and maximum 100 years old", min_value=18,
                          max_value=100, step=1, value=25)
    sex = st.selectbox("Gender: male or female", gender)
    bmi = st.number_input("BMI: Body Mass Index", min_value=15.0,
                          max_value=60.0, step=0.1, value=25.5)
    children = st.selectbox("Children: Number of Children", child)
    smoker = st.selectbox("Smoker", smoke)
    region = st.selectbox("Region", regions)

    pred_button = st.button("Predict Charges")

    if pred_button:
        # Convert inputs to appropriate data types
        inputs = np.array([[age, sex, bmi, children, smoker, region]])
        columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        df = pd.DataFrame(inputs, columns=columns)

        # Load the model
        try:
            with open('insurance_model.pkl', 'rb') as file:
                model = pickle.load(file)
        except FileNotFoundError:
            st.error("Model file not found.")
            return

        # Apply the same preprocessing as during training
        try:
            # Ensure the preprocessing step matches the pipeline used during training
            processed_df = model.named_steps['feature_engineering'].transform(
                df)
            processed_df = model.named_steps['preprocess'].transform(
                processed_df)

            # Make the prediction
            predicted_value = model.named_steps['model'].predict(processed_df)

            st.write(f'Predicted Charge: ${predicted_value[0]:,.2f}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")
