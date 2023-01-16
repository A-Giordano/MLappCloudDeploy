import pytest
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os
print("CWD", os.getcwd())
from src.ml.data import process_data
from src.ml.model import train_model, compute_model_metrics, inference, save_performance_on_slices

@pytest.fixture(scope="session")
def X_train_y_train_X_test_y_test_encoder_lb():
    data = pd.read_csv("data/census.csv")
    print(data.shape)
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    return X_train, y_train, X_test, y_test, encoder, lb

@pytest.fixture(scope="session")
def model():
    with open("model/inference_model.pkl", "rb") as input_file:
        model = pickle.load(input_file)
        return model


def test_train_model(X_train_y_train_X_test_y_test_encoder_lb):
    X_train, y_train, X_test, y_test, encoder, lb = X_train_y_train_X_test_y_test_encoder_lb
    cls = train_model(X_train, y_train)
    assert cls is not None

def test_inference(model, X_train_y_train_X_test_y_test_encoder_lb):
    X_train, y_train, X_test, y_test, encoder, lb = X_train_y_train_X_test_y_test_encoder_lb
    preds = inference(model, X_test)
    assert preds.shape == y_test.shape

def test_compute_model_metrics(model, X_train_y_train_X_test_y_test_encoder_lb):
    X_train, y_train, X_test, y_test, encoder, lb = X_train_y_train_X_test_y_test_encoder_lb
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert precision >= 0.6
    assert recall >= 0.6
    assert fbeta >= 0.6




