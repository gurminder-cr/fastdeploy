# ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. It is designed to run asynchronous web applications and APIs, making it a popular choice for frameworks like FastAPI, Starlette, and even Django when using its ASGI
# uvicorn pip install uvicorn 

import uvicorn 
import pandas as pd
import joblib 
import numpy as np
from fastapi import FastAPI # pip install fastapi 
                            # file_name: object_name
# to run fastapi app -- uvicorn app:app --reload 
# get, post, update, delete 
from inputData import heartInput
classifier= joblib.load('model.pkl')

app= FastAPI()

@app.get('/')
def index():
    return {"message":"Hello Coder Roots"}

@app.get('/name')
def printName():
    return {"message":"Hello Baljinder"}

@app.get('/{naam}')
def printNaam(naam:str):
    return {'Name is':f"{naam}"}

@app.post('/predict-disease')
def predictDisease(data:heartInput):
    data= data.model_dump()
    age=data['age']
    sex=data['sex']
    cp= data['cp']
    trestbps= data['trestbps'] 
    chol= data['chol']
    fbs = data['fbs']
    restecg= data['restecg']
    thalach= data['thalach']
    exang= data['exang']
    oldpeak=data['oldpeak']
    slope= data['slope']
    ca= data['ca']
    thal= data['thal']
    
    host_data= [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    pred_result = classifier.predict([host_data])
    
    if pred_result[0]==1:
        result="Heart Disease"
    else:
        result="No Heart Disease"
    
    return {
        'prediction':result
    }
    # print(host_data)

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)    
    
# post - heart disease model 

