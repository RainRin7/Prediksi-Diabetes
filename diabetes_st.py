from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle

import streamlit as st

# Loading the model
f = open("diabetes_dt.pckl", "rb")
dt_classifier = pickle.load(f)
f.close()

# User Interface
st.title("Pengecekan Dini Diabetes")

st.write("### Input Data")
col1, col2 = st.columns(2)
age = col1.number_input("Umur", min_value=0, max_value=100, value=45)
gender = col2.selectbox("Jenis Kelamin", ("Laki-Laki", "Perempuan"))

col1, col2, col3 = st.columns([2, 2, 1])
height = col1.number_input("Tinggi (cm)", min_value=0, max_value=300, value=165)
weight = col2.number_input("Berat Badan (kg)", min_value=0, max_value=150, value=100)
bmi = weight/(height*0.01)**2
col3.metric(label="BMI", value=f"{bmi:.2f}")

col1, col2 = st.columns(2)
glucose = col1.number_input("Kadar Glukosa", min_value=0.0, max_value=300.0, value=150.0)
bloodpressure = col1.number_input("Tekanan Darah (Diastolik)", min_value=0.0, max_value=300.0, value=50.0)
insulin = col2.number_input("Kadar Insulin", min_value=0.0, max_value=300.0, value=50.0)
pregnancies = col2.number_input("Usia Kehamilan (dalam bulan)", min_value=0, max_value=10, value=0, disabled=gender == "Laki-Laki")

if gender == "Laki-Laki":
  pregnancies = 0
new_case = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bloodpressure],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Age': [age]
})
prediction = dt_classifier.predict(new_case)
confidence = dt_classifier.predict_proba(new_case)
labels = ["Tidak Diabetes", "Diabetes"]

st.write("### Hasil Prediksi")
col1, col2 = st.columns([2, 1])
col1.header(f'{labels[prediction[0]]}')
col2.metric(label="Confidence", value=f"{max(confidence[0]) * 100:.2f}%")
