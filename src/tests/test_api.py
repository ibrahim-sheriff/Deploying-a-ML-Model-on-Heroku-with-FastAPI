import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_greetings():
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"
    assert response.json() == 'Greetings and salutations everybody'


@pytest.mark.parametrize('test_input, expected', [
    ('age', "Age of the person - numerical - int"),
    ('fnlgt', 'MORE INFO NEEDED - numerical - int'),
    ('race', 'Race of the person - nominal categorical - str')
])
def test_feature_info(test_input, expected):
    response = client.get(f'/feature_info/{test_input}')
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"
    assert response.json() == expected


def test_predict():
    data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"
    assert response.json()['label'] == 0 or response.json()['label'] == 1
    assert response.json()['prob'] >= 0 and response.json()['label'] <= 1
    assert response.json()['salary'] == '>50k' or response.json()[
        'salary'] == '<=50k'


def test_missing_feature_predict():
    data = {
        "age": 0
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.request.method == "POST"
    assert response.json()["detail"][0]["type"] == "value_error.missing"
