import streamlit as st
import pickle
import numpy as np	


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction")

st.write("Enter Patient Details")

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=1)


if st.button("Predict"):

    input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                            SkinThickness, Insulin, BMI,
                            DiabetesPedigreeFunction, Age]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("Diabetes Detected")
    else:
        st.success("No Diabetes Detected")