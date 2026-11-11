import json,os,time
from pathlib import Path
import joblib,pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel,Field
from prometheus_client import Counter,Histogram,generate_latest,CONTENT_TYPE_LATEST
from starlette.responses import Response
from .common import ROOT
app=FastAPI(title='API de probabilidad de renovación',version='1.0.0')
REQUESTS=Counter('prediction_requests_total','Total peticiones', ['status'])
LATENCY=Histogram('prediction_latency_seconds','Tiempo de prediccion')
class PredictInput(BaseModel):
    features:dict[str,float|int|str|None]=Field(description='Variables predictoras sin FLAG_VENTA/CLIENTE/MES')
model_bundle=None
def get_model():
    global model_bundle
    if model_bundle is None:
        path=Path(os.getenv('MODEL_PATH',str(ROOT/'artifacts/model.joblib')))
        if not path.exists(): raise HTTPException(status_code=503,detail='Modelo no disponible. Ejecuta el entrenamiento.')
        model_bundle=joblib.load(path)
    return model_bundle
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/ready')
def ready():
    get_model();return {'status':'ready'}
@app.post('/predict')
def predict(record:PredictInput):
    start=time.perf_counter()
    try:
        bundle=get_model()
        forbidden={'FLAG_VENTA','CLIENTE','MES'} & record.features.keys()
        if forbidden: raise HTTPException(status_code=422,detail=f'Campos prohibidos: {sorted(forbidden)}')
        missing=set(bundle['columns'])-set(record.features)
        extra=set(record.features)-set(bundle['columns'])
        if missing or extra: raise HTTPException(status_code=422,detail={'missing':sorted(missing),'extra':sorted(extra)})
        prob=float(bundle['pipeline'].predict_proba(pd.DataFrame([record.features],columns=bundle['columns']))[0,1])
        response={'probabilidad_renovacion':round(prob,6),'prediccion':int(prob>=bundle['threshold']),'umbral':bundle['threshold']}
        REQUESTS.labels(status='ok').inc()
        path=ROOT/'monitoring/predictions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f: f.write(json.dumps({'ts':time.time(),'probability':prob,'positive':response['prediccion']})+'\n')
        return response
    except HTTPException:
        REQUESTS.labels(status='error').inc();raise
    finally: LATENCY.observe(time.perf_counter()-start)
@app.get('/metrics')
def metrics(): return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
