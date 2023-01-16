from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def json_sample(request):
    payload = {
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
        "salary": ">50K"
    }

    return payload

@pytest.fixture(scope="session")
def json_sample_2(request):
    payload = {
        "age": 66,
        "workclass": "Private",
        "fnlgt": 211781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
        "salary": "<=50K"
    }

    return payload


@pytest.fixture(scope="session")
def json_sample_with_error(request):
    payload = {
        "age": 66,
        "workclass": "Private",
        "fnlgt": 211781,
        "education": "Masters",
        "education-num": "Fourteen", #14,
        "marital-status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
        "salary": "<=50K"
    }

    return payload


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to FastAPI interface to income classifier"}


def test_predict(client, json_sample):
    response = client.post("/predict", json=json_sample)
    assert response.status_code == 200
    assert response.json()["result"] == json_sample["salary"]


def test_predict_2(client, json_sample_2):
    response = client.post("/predict", json=json_sample_2)
    assert response.status_code == 200
    assert response.json()["result"] == json_sample_2["salary"]

def test_predict_error_422(client, json_sample_with_error):
    response = client.post("/predict", json=json_sample_with_error)
    assert response.status_code == 422