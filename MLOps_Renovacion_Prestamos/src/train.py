import argparse, json, os
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .common import ROOT, CATEGORICAL, load_data, split_temporal, features
def make_pipeline(x):
    cat=[c for c in CATEGORICAL if c in x.columns]
    num=[c for c in x.columns if c not in cat]
    pre=ColumnTransformer([
       ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
       ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])
    return Pipeline([('preprocessor',pre),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
def score(y,prob,threshold):
    pred=(prob>=threshold).astype(int)
    return {'average_precision':float(average_precision_score(y,prob)), 'roc_auc':float(roc_auc_score(y,prob)),
      'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)), 'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
      'prevalence':float(np.mean(y)),'n_samples':int(len(y))}
def fit(data_path,output_dir,tracking=False):
    d=load_data(data_path)
    tr,va,te=split_temporal(d)
    xtr=features(tr); xva=features(va); xte=features(te)
    model=make_pipeline(xtr)
    model.fit(xtr,tr.FLAG_VENTA)
    pval=model.predict_proba(xva)[:,1]
    # Umbral definido EXCLUSIVAMENTE sobre validación para evitar fuga de test.
    thresholds=np.linspace(.05,.95,91)
    best=max(thresholds,key=lambda t:f1_score(va.FLAG_VENTA,pval>=t,zero_division=0))
    ptest=model.predict_proba(xte)[:,1]
    out={'case':'Renovacion de prestamos','split':{'train':'201501-201507','validation':'201508','test':'201509'},
         'threshold':round(float(best),4),'train_rows':len(tr),'validation':score(va.FLAG_VENTA,pval,best),
         'test':score(te.FLAG_VENTA,ptest,best),'model':'LogisticRegression(class_weight=balanced)',
         'created_utc':datetime.now(timezone.utc).isoformat(),'input_columns':list(xtr.columns)}
    output_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':model,'threshold':float(best),'columns':list(xtr.columns),'reference_missing_rate':{c:float(xtr[c].isna().mean()) for c in xtr.columns}},output_dir/'model.joblib')
    (output_dir/'metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    if tracking:
      try:
       import mlflow
       mlflow.set_experiment('renovacion-prestamos')
       with mlflow.start_run():
        mlflow.log_params({'model':'LogisticRegression','split':'temporal','threshold':float(best)})
        mlflow.log_metrics({f'validation_{k}':v for k,v in out['validation'].items() if isinstance(v,float)})
        mlflow.log_metrics({f'test_{k}':v for k,v in out['test'].items() if isinstance(v,float)})
        mlflow.log_artifact(str(output_dir/'metrics.json'))
        mlflow.log_artifact(str(output_dir/'model.joblib'))
      except ImportError: raise RuntimeError('Instala las dependencias de requirements.txt para habilitar MLflow')
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(ROOT/'data/raw/Dataset Renovacion_prestamo.csv'))
    p.add_argument('--out',default=str(ROOT/'artifacts'))
    p.add_argument('--mlflow',action='store_true')
    a=p.parse_args()
    print(json.dumps(fit(__import__('pathlib').Path(a.data),__import__('pathlib').Path(a.out),a.mlflow),indent=2,ensure_ascii=False))
