# Put the code for your API here.

from pydantic import BaseModel
from fastapi import FastAPI
from typing import Optional
import pandas as pd
import joblib
from src.ml.data import process_data, decode_pred
from src.ml.model import inference


app = FastAPI()

model = joblib.load("model/inference_model.pkl")
encoder = joblib.load("model/encoder.pkl")
binarizer = joblib.load("model/label_binarizer.pkl")


@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to income classifier"
    }


def replace_dash(string: str) -> str:
    return string.replace('_', '-')


class Features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = replace_dash
        schema_extra = {
            "example": {
                "age": 31,
                "workclass": "Private",
                "fnlgt": 45781,
                "education": "Masters",
                "education-num": 14,
                "marital-status": "Never-married",
                "occupation": "Prof-speciality",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital-gain": 14000,
                "capital-loss": 0,
                "hours-per-week": 55,
                "native-country": "United-States",
            }
        }


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
        label=None, training=False, encoder=encoder, lb=binarizer)

    preds = decode_pred(binarizer, inference(model, X_train))[0]
    return {"result": preds}
