import os
import pandas as pd
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder


load_dotenv()

def get_data():

    url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(url)

    query = "select * from patients;"
    data = pd.read_sql(query, engine)
    engine.dispose()
     
    drop_col = ["encounter_id", "patient_nbr",
                "weight", "payer_code", "medical_specialty",
                "diag_1", "diag_2", "diag_3","acetohexamide", "tolbutamide",
                "troglitazone", "tolazamide", "examide", 
                "citoglipton", "glipizide-metformin", "glimepiride-pioglitazone", 
                "metformin-rosiglitazone", "metformin-pioglitazone"]

    data = data.drop(columns=drop_col)
    data = data[data["gender"] != 'Unknown/Invalid']
    

    data["max_glu_serum"] = data["max_glu_serum"].replace('?', 'None').fillna('None')
    data["A1Cresult"] = data["A1Cresult"].replace('?', 'None').fillna('None')
    data["race"] = data["race"].replace('?', 'Unknown')
    data['age'] = data['age'].map({'[0-10)':1,'[10-20)':2,'[20-30)':3,'[30-40)':4,
                                   '[40-50)':5,'[50-60)':6,'[60-70)':7,'[70-80)':8,
                                   '[80-90)':9,'[90-100)':10})


    medi_cols = ["metformin", "repaglinide", "nateglinide", 
                 "chlorpropamide", "glimepiride",
                 "glipizide", "glyburide",  "pioglitazone", 
                 "rosiglitazone", "acarbose", "miglitol", 
                 "insulin", "glyburide-metformin",]

    for i in medi_cols:
         data[i] = data[i].map({'No': 0,'Steady': 1,'Up': 2,'Down': 3})

    le_cols=["race", "gender", "change", "diabetesMed",
              "max_glu_serum", "A1Cresult", "admission_type_id", 
              "discharge_disposition_id", "admission_source_id"]
    
    
    
    for x in le_cols:
         le = LabelEncoder()
         data[x] = le.fit_transform(data[x])
         joblib.dump(le, f"model/{x}.pkl")
    l_en = LabelEncoder()
    data["readmitted"] = l_en.fit_transform(data["readmitted"])
    joblib.dump(l_en, f"model/target_encoder.pkl")

    X = data.drop(columns=['readmitted']) 
    y = data['readmitted']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    sc = StandardScaler()
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses']

    X_train[num_cols] = sc.fit_transform(X_train[num_cols])
    X_test[num_cols] = sc.transform(X_test[num_cols])

    joblib.dump(sc,"model/scaler.pkl")

    sm = SMOTE(random_state = 42)

    X_train,y_train  = sm.fit_resample(X_train,y_train)

    joblib.dump(X_train,"model/X_train.pkl")
    joblib.dump(X_test,"model/X_test.pkl")
    joblib.dump(y_train,"model/y_train.pkl")
    joblib.dump(y_test,"model/y_test.pkl")



    return X_train,X_test,y_train,y_test

    

get_data()