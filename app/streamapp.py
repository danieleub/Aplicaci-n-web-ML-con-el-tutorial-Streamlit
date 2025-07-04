import streamlit as st
import numpy as np
import pickle
import os

# Cargar modelo
modelo_path = os.path.join(os.path.dirname(__file__), 'boosting.sav')
modelo = pickle.load(open(modelo_path, 'rb'))

st.set_page_config(page_title="Predicción de Diabetes", layout="centered")

st.title("🔍 Predicción de diabetes")

# Formulario
with st.form("diabetes_form"):
    Pregnancies = st.number_input("Embarazos", min_value=0, step=1)
    Glucose = st.number_input("Glucosa", min_value=0, step=1)
    BloodPressure = st.number_input("Presión sanguínea", min_value=0, step=1)
    SkinThickness = st.number_input("Grosor de piel", min_value=0, step=1)
    Insulin = st.number_input("Insulina", min_value=0, step=1)
    BMI = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Función genética de diabetes", min_value=0.0, step=0.01)
    Age = st.number_input("Edad", min_value=0, step=1)

    submit = st.form_submit_button("Predecir")

if submit:
    try:
        data = [[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]]
        prediction = modelo.predict(np.array(data))[0]
        resultado = "✅ Tiene diabetes" if prediction == 1 else "❌ No tiene diabetes"
        st.success(f"Resultado: {resultado}")
    except Exception as e:
        st.error(f"Error: {e}")
