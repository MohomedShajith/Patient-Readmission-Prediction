import os
import io
import json
import requests
import sounddevice
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from groq import Groq
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

url = os.getenv("MONGO_URI")
client = MongoClient(url)
db = client["patientdb"]
collection = db["predictions"]

def record_and_transcribe():
    seconds = 5
    samplerate = 16000
    audio = sounddevice.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sounddevice.wait()
    audio = (audio.flatten() * 32768).astype(np.int16)
    buffer = io.BytesIO()
    wav.write(buffer, samplerate, audio)
    buffer.seek(0)
    buffer.name = "audio.wav"
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    transcription = groq_client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=buffer
    )
    return transcription.text

gender = {"Female": 0, "Male": 1}
race = {"AfricanAmerican": 0, "Asian": 1, "Caucasian": 2, "Hispanic": 3, "Other": 4, "Unknown": 5}
max_glu_serum = {">200": 0, ">300": 1, "None": 2, "Norm": 3}
A1Cresult = {">7": 0, ">8": 1, "None": 2, "Norm": 3}
change = {"Ch": 0, "No": 1}
diabetesMed = {"No": 0, "Yes": 1}
medication = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

st.title("Patient Readmission Prediction")

tab1, tab2, tab3 = st.tabs(["Prediction", "Text & Voice Search", "Analytics"])

with tab1:
    st.header("Patient Prediction")

    gender_value = gender[st.selectbox("Gender", list(gender.keys()))]
    race_value = race[st.selectbox("Race", list(race.keys()))]
    glucose_value = max_glu_serum[st.selectbox("Max Glucose Serum", list(max_glu_serum.keys()))]
    a1c_value = A1Cresult[st.selectbox("A1C Result", list(A1Cresult.keys()))]
    change_value = change[st.selectbox("Change in Medication", list(change.keys()))]
    diabetes_value = diabetesMed[st.selectbox("Diabetes Medication", list(diabetesMed.keys()))]

    metformin_value = medication[st.selectbox("Metformin", list(medication.keys()))]
    repaglinide_value = medication[st.selectbox("Repaglinide", list(medication.keys()))]
    nateglinide_value = medication[st.selectbox("Nateglinide", list(medication.keys()))]
    chlorpropamide_value = medication[st.selectbox("Chlorpropamide", list(medication.keys()))]
    glimepiride_value = medication[st.selectbox("Glimepiride", list(medication.keys()))]
    glipizide_value = medication[st.selectbox("Glipizide", list(medication.keys()))]
    glyburide_value = medication[st.selectbox("Glyburide", list(medication.keys()))]
    pioglitazone_value = medication[st.selectbox("Pioglitazone", list(medication.keys()))]
    rosiglitazone_value = medication[st.selectbox("Rosiglitazone", list(medication.keys()))]
    acarbose_value = medication[st.selectbox("Acarbose", list(medication.keys()))]
    miglitol_value = medication[st.selectbox("Miglitol", list(medication.keys()))]
    insulin_value = medication[st.selectbox("Insulin", list(medication.keys()))]
    glyburide_metformin_value = medication[st.selectbox("Glyburide-Metformin", list(medication.keys()))]

    age = st.slider("Age Group (1=0-10yrs, 10=90-100yrs)", 1, 10, 5)
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_lab_procedures = st.slider("Number of Lab Procedures", 1, 132, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_medications = st.slider("Number of Medications", 1, 81, 15)
    number_outpatient = st.slider("Number of Outpatient Visits", 0, 42, 0)
    number_emergency = st.slider("Number of Emergency Visits", 0, 76, 0)
    number_inpatient = st.slider("Number of Inpatient Visits", 0, 21, 0)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    admission_type_id = st.slider("Admission Type ID", 1, 8, 1)
    discharge_disposition_id = st.slider("Discharge Disposition ID", 1, 26, 1)
    admission_source_id = st.slider("Admission Source ID", 1, 25, 7)

    if st.button("Predict"):
        payload = {
            "race": race_value,
            "gender": gender_value,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": float(time_in_hospital),
            "num_lab_procedures": float(num_lab_procedures),
            "num_procedures": float(num_procedures),
            "num_medications": float(num_medications),
            "number_outpatient": float(number_outpatient),
            "number_emergency": float(number_emergency),
            "number_inpatient": float(number_inpatient),
            "number_diagnoses": float(number_diagnoses),
            "max_glu_serum": glucose_value,
            "A1Cresult": a1c_value,
            "metformin": metformin_value,
            "repaglinide": repaglinide_value,
            "nateglinide": nateglinide_value,
            "chlorpropamide": chlorpropamide_value,
            "glimepiride": glimepiride_value,
            "glipizide": glipizide_value,
            "glyburide": glyburide_value,
            "pioglitazone": pioglitazone_value,
            "rosiglitazone": rosiglitazone_value,
            "acarbose": acarbose_value,
            "miglitol": miglitol_value,
            "insulin": insulin_value,
            "glyburide_metformin": glyburide_metformin_value,
            "change": change_value,
            "diabetesMed": diabetes_value
        }

        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["Patient_Readmission_Prediction"]
            probability = result["Patient_Readmission_Probability"]

            if prediction == "<30":
                st.error(f"⚠️ HIGH RISK — Patient likely to be readmitted within 30 days | Probability: {probability}")
            elif prediction == ">30":
                st.warning(f"⚠️ MODERATE RISK — Patient may be readmitted after 30 days | Probability: {probability}")
            else:
                st.success(f"✅ LOW RISK — Patient unlikely to be readmitted | Probability: {probability}")
        else:
            st.error("API Error — make sure FastAPI is running")

with tab2:
    st.header("Voice / Text Search")

    query = st.text_input("Ask a question about patient predictions...")

    col1, col2 = st.columns(2)
    with col1:
        search_clicked = st.button("Search")
    with col2:
        voice_clicked = st.button("🎤 Record & Search")

    if voice_clicked:
        st.info("Recording for 5 seconds... Speak now!")
        query = record_and_transcribe()
        st.success(f"You said: {query}")

    if search_clicked or voice_clicked:
        filter_prompt = f"""
Convert this question to a MongoDB filter JSON only.
Collection fields: prediction ("<30", ">30", "NO"), probability (float), timestamp
Return ONLY valid JSON, nothing else. No explanation, no markdown, no backticks.
Examples:
"show high risk patients" -> {{"prediction": "<30"}}
"show low risk patients" -> {{"prediction": "NO"}}
"show moderate risk patients" -> {{"prediction": ">30"}}
"probability above 0.7" -> {{"probability": {{"$gt": 0.7}}}}
"all patients" -> {{}}

Question: {query}
"""
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        filter_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": filter_prompt}]
        )

        filter_text = filter_response.choices[0].message.content.strip()

        try:
            mongo_filter = json.loads(filter_text)
        except:
            mongo_filter = {}

        filtered_data = list(collection.find(mongo_filter, {"_id": 0}))

        answer_prompt = f"""
You are a medical data assistant analyzing patient readmission predictions.
- "<30" means HIGH RISK (readmitted within 30 days)
- ">30" means MODERATE RISK (readmitted after 30 days)
- "NO" means LOW RISK (not readmitted)

Here is the filtered patient data:
{filtered_data}

Answer this question concisely:
{query}
"""
        answer_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": answer_prompt}]
        )

        st.write(answer_response.choices[0].message.content)
        st.code(f"MongoDB Filter: {json.dumps(mongo_filter, indent=2)}")
        st.dataframe(filtered_data)

with tab3:
    st.header("Analytics")

    results = collection.find({}, {"_id": 0})
    data = list(results)
    df = pd.DataFrame(data)

    fig = px.pie(df, names='prediction', title='Patient Readmission Distribution')
    st.plotly_chart(fig)

    bar = px.bar(
        df["prediction"].value_counts().reset_index(),
        x="prediction",
        y="count",
        color="prediction",
        title="Count of Prediction Classes"
    )
    st.plotly_chart(bar)

    hist = px.histogram(
        df,
        x="probability",
        nbins=10,
        color_discrete_sequence=["#EF553B"],
        title="Probability Score Distribution"
    )
    st.plotly_chart(hist)