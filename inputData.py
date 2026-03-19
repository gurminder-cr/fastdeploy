# pip install pydantic - pydantic is used to fetch data from the host
 
from pydantic import BaseModel
class heartInput(BaseModel):
    age:float
    sex:int
    cp:int
    trestbps:int 
    chol:int 
    fbs:int
    restecg:int 
    thalach: float
    exang:int 
    oldpeak:float
    slope:int
    ca:int
    thal:int 