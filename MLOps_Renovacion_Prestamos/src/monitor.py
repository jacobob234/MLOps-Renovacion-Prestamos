import argparse,json
from pathlib import Path
import pandas as pd
from .common import ROOT,load_data,features
def report(data_path,model_path,output):
    import joblib
    bundle=joblib.load(model_path)
    df=load_data(data_path)
    current=features(df)
    missing=[]
    for c in bundle['columns']:
        reference=bundle['reference_missing_rate'][c]
        observed=float(current[c].isna().mean())
        if abs(observed-reference)>.1: missing.append({'feature':c,'reference_missing':reference,'current_missing':observed})
    result={'n_records':len(df),'missingness_alerts':missing,'status':'alert' if missing else 'ok',
       'note':'Monitor básico de calidad. Desempeño real requiere etiquetas post-despliegue.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True)
    p.add_argument('--model',default=str(ROOT/'artifacts/model.joblib'))
    p.add_argument('--out',default=str(ROOT/'monitoring/report.json'))
    args=p.parse_args();print(report(Path(args.data),Path(args.model),Path(args.out)))
