









import requests

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

res = requests.post("https://mlappclouddeploy.onrender.com/predict", json=payload)
print('status_code', res.status_code)
print('json', res.json())
