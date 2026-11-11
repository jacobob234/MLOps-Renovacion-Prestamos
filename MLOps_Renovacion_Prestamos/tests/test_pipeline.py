import pandas as pd
from src.common import split_temporal,features
from src.train import make_pipeline
def test_split_temporal():
    d=pd.DataFrame({'MES':[201501,201508,201509],'FLAG_VENTA':[0,1,0],'CLIENTE':[1,2,3]})
    a,b,c=split_temporal(d)
    assert len(a)==len(b)==len(c)==1
def test_no_leakage():
    d=pd.DataFrame({'MES':[201501],'CLIENTE':[101],'FLAG_VENTA':[1],'EDAD':[30]})
    assert list(features(d).columns)==['EDAD']
def test_training_pipeline():
    x=pd.DataFrame({'EDAD':[25,31,48,52,22,38],'SEXO':['M','F','M','F','M','F'],'REGION':['A','B','A','B','A','B']})
    y=[0,0,1,1,0,1]
    model=make_pipeline(x).fit(x,y)
    assert len(model.predict_proba(x))==6
