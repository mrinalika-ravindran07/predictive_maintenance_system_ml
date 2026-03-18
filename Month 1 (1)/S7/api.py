from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 1. Initialize App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all connections
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Train Model in Memory 
print("Training model in-memory to ensure compatibility...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained and ready!")

# --- 4. Input Schema ---
class PatientData(BaseModel):
    age: float
    bmi: float

# --- 5. Prediction Endpoint ---
@app.post("/predict")
def predict(data: PatientData):
    # Normalize inputs (approximating the scaling used in training)
    # Age (Mean: ~48.5, Std: ~13.1)
    normalized_age = (data.age - 48.5) / 13.1 * 0.04  # Scaling factor adjustment
    
    # BMI (Mean: ~26.3, Std: ~4.4)
    normalized_bmi = (data.bmi - 26.3) / 4.4 * 0.04   # Scaling factor adjustment

    # Create Feature Array (10 features total)
    # Order: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
    features = np.zeros((1, 10))
    features[0, 0] = normalized_age
    features[0, 2] = normalized_bmi
    
    # Predict
    prediction = model.predict(features)
    
    return {"disease_progression": float(prediction[0])}

# --- 6. Homepage Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Server is running perfectly!"}