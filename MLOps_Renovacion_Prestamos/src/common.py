from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TARGET = 'FLAG_VENTA'
EXCLUDE = ['FLAG_VENTA','CLIENTE','MES']
CATEGORICAL = ['SEXO','EST_CIVIL','REGION']
def load_data(path):
    df = pd.read_csv(path,sep=';',encoding='latin1')
    required = {'MES','CLIENTE',TARGET,*CATEGORICAL}
    missing = required-set(df.columns)
    if missing: raise ValueError(f'Columnas faltantes: {sorted(missing)}')
    if not df[TARGET].dropna().isin([0,1]).all(): raise ValueError('FLAG_VENTA debe ser binaria')
    if df[TARGET].isna().any(): raise ValueError('Etiquetas vacías')
    return df
def split_temporal(df):
    train=df[df.MES<=201507].copy()
    valid=df[df.MES==201508].copy()
    test=df[df.MES>=201509].copy()
    if min(map(len,(train,valid,test)))==0: raise ValueError('Faltan periodos de train/val/test')
    return train,valid,test
def features(df): return df.drop(columns=EXCLUDE,errors='ignore')
