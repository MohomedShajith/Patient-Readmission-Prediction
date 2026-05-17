import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient


load_dotenv()
app =FastAPI()

url = os.getenv("MONGO_URI")
client = MongoClient(url)
db = client["patientdb"]
collection = db["predictions"]




class PatientData(BaseModel):
    race:int
    gender:int
    age :int
    admission_type_id  :int
    discharge_disposition_id:int
    admission_source_id:int
    time_in_hospital  :float
    num_lab_procedures:float
    num_procedures    :float
    num_medications   :float
    number_outpatient :float
    number_emergency  :float
    number_inpatient  :float
    number_diagnoses  :float
    max_glu_serum:int
    A1Cresult:int
    metformin :int
    repaglinide:int
    nateglinide:int
    chlorpropamide:int
    glimepiride:int
    glipizide :int
    glyburide :int
    pioglitazone :int
    rosiglitazone:int
    acarbose:int
    miglitol:int
    insulin:int
    glyburide_metformin:int
    change:int
    diabetesMed:int


xgb_model = joblib.load("model/xg_model.pkl")
scaler = joblib.load("model/scaler.pkl")
target_en = joblib.load("model/target_encoder.pkl")



@app.post("/predict")
def predict(patientdata: PatientData):
    df = pd.DataFrame([patientdata.dict()])
    df = df.rename(columns={"glyburide_metformin": "glyburide-metformin"})

    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses']

    df[num_cols] = scaler.transform(df[num_cols])
    
    xg_proba = xgb_model.predict_proba(df)
    xg_threshold = 0.20

    xg_tuned_preds = np.argmax(xg_proba, axis=1)
    xg_tuned_preds[xg_proba[:, 0] > xg_threshold] = 0

    prediction = target_en.inverse_transform(xg_tuned_preds)

    collection.insert_one({"patient":patientdata.dict(),
    "prediction": prediction[0],
    "probability": round(float(xg_proba[0][xg_tuned_preds[0]]), 2),
     "timestamp":datetime.now()})
    
    return {"Patient_Readmission_Prediction":prediction[0],"Patient_Readmission_Probability":round(float(xg_proba[0][xg_tuned_preds[0]]), 2)}




@app.get("/health")
def health_check():
    return {"status": "ok"}



@app.get("/history")
def history_preds():
    results = collection.find({}, {"_id": 0})
    return list(results)



