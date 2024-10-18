#importing neccessary libraries #http://127.0.0.1:8000/
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Define the input data model
class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# Load the saved model
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))


# Serve the HTML page to get and read the HTMl page
@app.get("/", response_class=HTMLResponse)
def get_ui():
    with open("template.html", "r") as file:
        return file.read()


# Predict endpoint, input data to predict
@app.post("/predict/")
def predict_diabetes(data: model_input):
    data = data.dict()
    Pregnancies = data['Pregnancies']
    Glucose = data['Glucose']
    BloodPressure = data['BloodPressure']
    SkinThickness = data['SkinThickness']
    Insulin = data['Insulin']
    BMI = data['BMI']
    DiabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    Age = data['Age']

    # Make the prediction using the loaded model
    prediction = diabetes_model.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Return prediction result
    if prediction[0] == 1:
        result = "Diabetes"
    else:
        result = "No Diabetes"

    return {"prediction": result}