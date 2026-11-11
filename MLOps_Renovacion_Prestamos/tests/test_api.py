from fastapi.testclient import TestClient
from src.api import app
def test_health():
    r=TestClient(app).get('/health')
    assert r.status_code==200 and r.json()['status']=='ok'
def test_invalid_payload():
    r=TestClient(app).post('/predict',json={'features':{'FLAG_VENTA':1}})
    assert r.status_code in (422,503)
