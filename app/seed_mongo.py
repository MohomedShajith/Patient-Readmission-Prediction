import os
import random
from faker import Faker
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

fake = Faker()
url = os.getenv("MONGO_URI")
client = MongoClient(url)

db = client["patientdb"]
collection = db["predictions"]

# Clear existing fake data first
collection.delete_many({"patient": None})

data = []

for _ in range(30):
    patient = {
        "patient_name": fake.name(),
        "prediction": random.choice(["<30", ">30", "NO"]),
        "probability": round(random.uniform(0.3, 0.9), 2),
        "timestamp": datetime.now()
    }
    data.append(patient)

collection.insert_many(data)
print("30 fake patient predictions inserted successfully")