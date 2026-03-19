# ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. It is designed to run asynchronous web applications and APIs, making it a popular choice for frameworks like FastAPI, Starlette, and even Django when using its ASGI
# uvicorn pip install uvicorn 
# to run - uvicorn mainaapi:api --reload

import uvicorn
import pandas as pd 
import numpy as np 
from fastapi import FastAPI 
from inputData import heartInput
import joblib
app= FastAPI()

model= joblib.load('model.pkl')
@app.get('/')
def index():
    return {"message":'Hello Class how are you'}

@app.get('/name')
def indexName():
    return {"message":'Hello Nikhil'}

@app.get('/{name}')
def printValue(name:str):
    return {"message":f'Hello {name}'}

@app.post('/predict')
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
    
    values=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

    ans= model.predict([values]) # [1] or [0]
    if ans[0]==1:
        result="Heart Disease"
    else:
        result="No Heart Disease"
    
    return {
        'prediction':result
    }

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1')    
    