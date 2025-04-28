from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Education Inequality Prediction API")

# Load model
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_transform.pkl")
rfr_model = joblib.load("rfr_model.pkl")
kmeans_final = joblib.load("kmeans_model.pkl")

class InputData(BaseModel):
    OOSR_Pre0Primary_Age_Male: float
    OOSR_Pre0Primary_Age_Female: float
    OOSR_Primary_Age_Male: float
    OOSR_Primary_Age_Female: float
    OOSR_Lower_Secondary_Age_Male: float
    OOSR_Lower_Secondary_Age_Female: float
    OOSR_Upper_Secondary_Age_Male: float
    OOSR_Upper_Secondary_Age_Female: float
    Completion_Rate_Primary_Male: float
    Completion_Rate_Primary_Female: float
    Completion_Rate_Lower_Secondary_Male: float
    Completion_Rate_Lower_Secondary_Female: float
    Completion_Rate_Upper_Secondary_Male: float
    Completion_Rate_Upper_Secondary_Female: float
    Grade_2_3_Proficiency_Reading: float
    Grade_2_3_Proficiency_Math: float
    Primary_End_Proficiency_Reading: float
    Primary_End_Proficiency_Math: float
    Lower_Secondary_End_Proficiency_Reading: float
    Lower_Secondary_End_Proficiency_Math: float
    Youth_15_24_Literacy_Rate_Male: float
    Youth_15_24_Literacy_Rate_Female: float
    Birth_Rate: float
    Gross_Primary_Education_Enrollment: float
    Gross_Tertiary_Education_Enrollment: float
    Unemployment_Rate: float

def preprocess_input(data: InputData):
    df = pd.DataFrame([data.dict()])
    df_scaled = scaler.transform(df)
    df_pca = pca.transform(df_scaled)
    return df_scaled, df_pca

@app.get("/")
def read_root():
    return {"message": "Education Inequality Prediction API is running"}

@app.post("/predict")
def predict_output(data: InputData):
    df_scaled, df_pca = preprocess_input(data)
    
    reg_prediction = rfr_model.predict(df_scaled)[0]
    cluster_prediction = kmeans_final.predict(df_pca)[0]
    
    return {
        "predicted_value": reg_prediction,
        "cluster_label": int(cluster_prediction)
    }