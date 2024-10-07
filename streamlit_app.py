import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_random_forest_model.pkl")

# Define expected feature columns based on the training set
expected_columns = [
    "Age", "Gender", "Biomass_Fuel_Exposure", "Occupational_Exposure", "Family_History_COPD",
    "BMI", "Air_Pollution_Level", "Respiratory_Infections_Childhood", "Pollution_Risk_Score",
    "Smoking_Status_encoded", "Smoking_Pollution_interaction",
    "Location_Biratnagar","Location_Birgunj","Location_Dharan","Location_Kathmandu","Location_Lalitpur","Location_Pokhara","Location_Rupandehi",
]

st.title("COPD Prediction Application")
st.write("Enter the input values to predict COPD Diagnosis")

# Collect user input for each feature
Age = st.number_input("Age", min_value=0, max_value=120, value=30)
Gender = st.selectbox("Gender (0 = Male, 1 = Female)", [0, 1])
Biomass_Fuel_Exposure = st.selectbox("Biomass Fuel Exposure", [0, 1])
Occupational_Exposure = st.selectbox("Occupational Exposure", [0, 1])
Family_History_COPD = st.selectbox("Family History of COPD", [0, 1])
BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
Air_Pollution_Level = st.number_input(
    "Air Pollution Level", min_value=0, max_value=500, value=100)
Respiratory_Infections_Childhood = st.selectbox(
    "Respiratory Infections in Childhood", [0, 1])
Pollution_Risk_Score = st.number_input(
    "Pollution Risk Score", min_value=0.0, max_value=100.0, value=0.0)
Smoking_Status_encoded = st.slider(
    "Smoking Status (0 = Non-smoker, 1 = Smoker)", 0.0, 1.0, 0.5)
Smoking_Pollution_interaction = st.number_input(
    "Smoking Pollution interaction", min_value=0.0, max_value=500.0, value=0.0)

# Categorical Location Input
location = st.selectbox("Location", ["Kathmandu","Pokhara","Rupandehi","Biratnagar","Birgunj","Lalitpur","Dharan"])

# Create a DataFrame with the input values
input_data = pd.DataFrame([{
    "Age": Age,
    "Gender": Gender,
    "Biomass_Fuel_Exposure": Biomass_Fuel_Exposure,
    "Occupational_Exposure": Occupational_Exposure,
    "Family_History_COPD": Family_History_COPD,
    "BMI": BMI,
    "Air_Pollution_Level": Air_Pollution_Level,
    "Respiratory_Infections_Childhood": Respiratory_Infections_Childhood,
    "Pollution_Risk_Score": Pollution_Risk_Score,
    "Smoking_Status_encoded": Smoking_Status_encoded,
    "Smoking_Pollution_interaction": Smoking_Pollution_interaction,
    "Location_Biratnagar": location == "Biratnagar",
    "Location_Birgunj": location == "Birgunj",
    "Location_Dharan": location == "Dharan",
    "Location_Kathmandu": location == "Kathmandu",
    "Location_Lalitpur": location == "Lalitpur",
    "Location_Pokhara": location == "Pokhara",
    "Location_Rupandehi": location == "Rupandehi",
 }])

# Align input DataFrame with expected columns
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Optional: Display the input data for debugging purposes
if st.checkbox("Show Input Data"):
    st.write(input_data)

# Predict using the loaded model
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(
        f"Predicted COPD Diagnosis: {'Positive' if prediction[0] == 1 else 'Negative'}")
