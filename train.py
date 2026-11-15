"""Reproduce the Cleveland model and held-out evaluation. No test-set tuning."""
import json, zipfile, urllib.request, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
ROOT=Path(__file__).parent
cols=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
path=ROOT/'data/processed.cleveland.data'
if not path.exists():
 with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/45/heart+disease.zip') as response:
  import io
  with zipfile.ZipFile(io.BytesIO(response.read())) as z: path.write_bytes(z.read('processed.cleveland.data'))
df=pd.read_csv(path,names=cols,na_values='?'); X=df.drop(columns='num'); y=(df.num>0).astype(int)
tr,te=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
nums=['age','trestbps','chol','thalach','oldpeak']; cats=[c for c in X if c not in nums]
pre=ColumnTransformer([('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
model=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=2000,random_state=42))]); model.fit(X.iloc[tr],y.iloc[tr])
p=model.predict_proba(X.iloc[te])[:,1]; pred=p>=.5
base=DummyClassifier(strategy='most_frequent').fit(X.iloc[tr],y.iloc[tr]); bp=base.predict(X.iloc[te])
def metrics(a,b):
 tn,fp,fn,tp=confusion_matrix(a,b,labels=[0,1]).ravel()
 return dict(n=len(a),accuracy=accuracy_score(a,b),recall=recall_score(a,b,zero_division=0),precision=precision_score(a,b,zero_division=0),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
num=pre.named_transformers_['num'];cat=pre.named_transformers_['cat']; co=model.named_steps['lr'].coef_[0];features=[];i=0
for j,name in enumerate(nums):
 features.append(dict(name=name,kind='numeric',fill=float(num['fill'].statistics_[j]),mean=float(num['scale'].mean_[j]),scale=float(num['scale'].scale_[j]),coef=float(co[i])));i+=1
for j,name in enumerate(cats):
 values=cat['onehot'].categories_[j].tolist();features.append(dict(name=name,kind='category',fill=float(cat['fill'].statistics_[j]),values=values,coef=co[i:i+len(values)].tolist()));i+=len(values)
records=[]
for k,idx in enumerate(te):
 values={c:(None if pd.isna(v) else float(v)) for c,v in X.iloc[idx].items()}
 records.append(dict(id=f'Test {k+1:02}',values=values,actual=int(y.iloc[idx]),probability=float(p[k])))
result=dict(features=features,intercept=float(model.named_steps['lr'].intercept_[0]),metrics=metrics(y.iloc[te],pred),auc=roc_auc_score(y.iloc[te],p),baseline=metrics(y.iloc[te],bp),test=records,train_n=len(tr),test_n=len(te),seed=42,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),versions={'sklearn':__import__('sklearn').__version__})
(ROOT/'model.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:result[k] for k in ['metrics','auc','baseline','train_n','test_n']},indent=2))
