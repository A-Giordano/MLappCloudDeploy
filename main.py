# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
from src.ml.data import process_data, decode_pred
from src.ml.model import inference


app = FastAPI()

model = joblib.load("model/inference_model.pkl")
encoder = joblib.load("model/encoder.pkl")
binarizer = joblib.load("model/label_binarizer.pkl")

# Home site with welcome message - GET request


@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to income classifier"
    }


# Alias Generator funtion for class CensusData
def replace_dash(string: str) -> str:
    return string.replace('_', '-')

# Class definition of the data that will be provided as POST request


class Features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int  # = Field(..., alias='education-num')
    marital_status: str  # = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int  # = Field(..., alias='capital-gain')
    capital_loss: int  # = Field(..., alias='capital-loss')
    hours_per_week: int  # = Field(..., alias='hours-per-week')
    native_country: str  # = Field(..., alias='native-country')
    salary: Optional[str]

    class Config:
        alias_generator = replace_dash


# POST request to /predict site. Used to validate model with sample census data
@app.post('/predict')
async def predict(input: Features):
    """
    POST request that will provide sample census data and expect a prediction
    Output:
        0 or 1
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])

    X_train, _, _, _ = process_data(
        input_df, categorical_features=cat_features,
        label='salary', training=False, encoder=encoder, lb=binarizer)

    preds = decode_pred(binarizer, inference(model, X_train))[0]
    return {"result": preds}
