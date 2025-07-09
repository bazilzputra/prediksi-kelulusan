import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder
model = pickle.load(open('model_kelulusan.sav', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
le_status = pickle.load(open('label_status.pkl', 'rb'))

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")
st.title("🎓 Prediksi Tingkat Kelulusan Mahasiswa")

# Form Input Data
with st.form("input_form"):
    gender = st.selectbox("Gender", ['female', 'male'])
    race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
    education = st.selectbox("Parental Level of Education", [
        "associate's degree", "bachelor's degree", "master's degree",
        "some college", "some high school", "high school"
    ])
    lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
    prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
    
    math = st.slider("Math Score", 0, 100, 60)
    reading = st.slider("Reading Score", 0, 100, 60)
    writing = st.slider("Writing Score", 0, 100, 60)

    submitted = st.form_submit_button("Prediksi")

# Prediksi jika tombol diklik
if submitted:
    input_df = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race,
        'parental level of education': education,
        'lunch': lunch,
        'test preparation course': prep,
        'math score': math,
        'reading score': reading,
        'writing score': writing
    }])
    
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    pred = model.predict(input_df)
    status = le_status.inverse_transform(pred)[0]

    if status == "Lulus":
        st.success("✅ Mahasiswa dinyatakan **LULUS**.")
    else:
        st.error("❌ Mahasiswa dinyatakan **TIDAK LULUS**.")
