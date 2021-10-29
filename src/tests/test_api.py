from fastapi.testclient import TestClient

from app import app


client = TestClient(app)

def test_greetings():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == 'Greetings and salutations everybody'
    
def test_feature_info():
    response = client.get('/feature_info/age')
    assert response.status_code == 200
    assert response.json() == "Age of the person - numerical - int"
    