import pandas as pd
import numpy as np
import joblib as jb
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
rfc = jb.load('C:/GFG_TITANIC_PROJECT/RFC_MODEL')
sc = jb.load('C:/GFG_TITANIC_PROJECT/RFC_SCALAR')

class titanic(BaseModel):
    Pclass : int
    female : int
    Age : int
    FamilySize : int
    IsAlone : int
    Fare : float

app = FastAPI(title = "TITANIC_DATASET")

@app.post("/predict_survivability",response_class= PlainTextResponse)
def titanic_prediction(user_inp : titanic):
    inputs = pd.DataFrame(
        {
            'Pclass' :	user_inp.Pclass,
            'Sex_Female' :  user_inp.female,
            'Age'    :  user_inp.Age,
            'FamilySize' : user_inp.FamilySize,
            'IsAlone' : user_inp.IsAlone,
            'Fare' : user_inp.Fare
        }, index = [0]
    )
    sc.transform(inputs)
    predict = rfc.predict(inputs)
    if predict[0] == 1 :
        predicted = "Survive"
    else :
        predicted = "Can't Survive"
    return predicted