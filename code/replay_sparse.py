"""Fixed original X: sparse GLS recomputation at previously fitted parameters."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import solveh_banded,cholesky_banded,cho_factor,cho_solve
from scipy.special import ndtr
from concurrent.futures import ProcessPoolExecutor
P=Path(__file__).resolve().parents[1];O=P/'replay_output';O.mkdir(exist_ok=True);D=O/'input';D.mkdir(exist_ok=True);sys.path.insert(0,str(D))
from kalman_core import states
H=[1,3,6,12];models=['M0','M1','M2d','M2dw']
def sparse_gls(W,ix,rho,nu):
 ph=rho**np.diff(ix);q=1-ph**2;d0=np.r_[1.,np.zeros(len(ix)-1)];d0[:-1]+=ph**2/q;d0[1:]+=1/q;off=-ph/q
 def Q(B):
  C=d0[:,None]*B;C[:-1]+=off[:,None]*B[1:];C[1:]+=off[:,None]*B[:-1];return C
 B=Q(W);ld=np.log(q).sum()
 if np.any(nu>0):
  d=np.sqrt(nu);F=np.zeros((2,len(ix)));F[0]=1+nu*d0;F[1,:-1]=off*d[:-1]*d[1:];ld+=2*np.log(cholesky_banded(F,lower=True,check_finite=False)[0]).sum();B=B-Q(d[:,None]*solveh_banded(F,d[:,None]*B,lower=True,check_finite=False))
 C=W.T@B;G=(C[1:,1:]+C[1:,1:].T)/2;b=cho_solve(cho_factor(G,check_finite=False),C[1:,0],check_finite=False);s=(W[:,0]-W[:,1:]@b)@(B[:,0]-B[:,1:]@b)/len(ix)
 return b,s,G,len(ix)*np.log(s)+ld

def run(zone):
 from pairwise_core import get_data,choose_station,make_table,build_design
 df,temp=get_data(D,[zone]);df=df[zone];station=choose_station(df,temp,'auto');table=make_table(df,temp,station,trend_rule='row');train=table[table.index.year<2007];test=table[table.index.year==2007];X,V,_,_=build_design(train,test)
 dt=pd.date_range('2005-01-01','2007-12-31 23:00',freq='h');N0=dt.searchsorted(pd.Timestamp('2007-01-01'));loc=dt.get_indexer(table.index);mask=np.zeros(len(dt),bool);mask[loc]=True;W=np.zeros((len(dt),286));W[loc,0]=table.load_mw;W[loc,1:]=np.vstack([X,V]);tt=np.arange(len(dt));Z=np.column_stack([np.sin(2*np.pi*tt/24),np.cos(2*np.pi*tt/24),np.sin(2*np.pi*tt/168),np.cos(2*np.pi*tt/168)])
 old=pd.read_csv(P/f'results/forecasts/zone_{zone:02d}.csv.gz');orig=dt.get_indexer(pd.to_datetime(old[(old.model=='M0')&(old.h==1)].origin))+1;assert (orig>0).all();params=pd.read_csv(P/'results/fitted_parameters.csv');params=params[params.zone==zone];rows=[];checks=[];diag=[]
 def score(model,h,mu,var):
  tar=orig+h-1;y=W[tar,0];sd=np.sqrt(np.maximum(var,1e-12));e=y-mu;z=e/sd;crps=sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi));width=1.6448536269514722*sd;o=old[(old.model==model)&(old.h==h)];assert len(o)==len(y) and np.max(np.abs(o.actual_MW.to_numpy()-y))<1e-8
  rows.append(dict(zone=zone,station=station,model=model,h=h,n=len(y),RMSE_MW=np.sqrt(np.mean(e*e)),MAE_MW=np.abs(e).mean(),CRPS_MW=crps.mean(),coverage90=np.mean(np.abs(e)<=width),interval_score90_MW=np.mean(2*width+20*np.maximum(np.abs(e)-width,0))))
  checks.append(dict(zone=zone,model=model,h=h,max_mean_diff=np.max(np.abs(mu-o.forecast_MW.to_numpy())),max_sd_diff=np.max(np.abs(sd-o.pred_sd_MW.to_numpy())),max_crps_diff=np.max(np.abs(crps-o.CRPS_MW.to_numpy()))))
 ix=np.flatnonzero(mask[:N0]);wc=W[ix]
 for p in params.itertuples():
  dim={'M0':0,'M1':0,'M2d':2,'M2dw':4}[p.model];g=np.array([getattr(p,f'gamma{i+1}') for i in range(dim)]);z=np.ascontiguousarray(Z[:,:dim]-Z[ix,:dim].mean(0));nu=p.a*np.exp(z@g);b,s,G,obj=sparse_gls(wc,ix,p.rho,nu[ix]);st,pv=states(W,mask,p.rho,p.a,g,z);st=st[orig-1];state=st[:,0]-st[:,1:]@b;cf=cho_factor(G,check_finite=False)
  diag.append(dict(zone=zone,model=p.model,sparse_objective=obj,kalman_objective=p.objective,abs_objective_diff=abs(obj-p.objective),sparse_scale_MW2=s,kalman_scale_MW2=p.scale_MW2))
  for h in H:
   tar=orig+h-1;fX=W[tar,1:];ph=p.rho**h;mu=fX@b+ph*state;A=fX-ph*st[:,1:];var=s*(1-ph*ph*(1-pv[orig-1])+nu[tar]+np.sum(A*cho_solve(cf,A.T,check_finite=False).T,axis=1));score(p.model,h,mu,var)
 return rows,checks,diag
if __name__=='__main__':
 with zipfile.ZipFile(P/'data/raw_inputs.zip') as z:
  for f in ['Load_history.csv','temperature_history.csv']:z.extract(f,D)
 states(np.ones((3,2)),np.ones(3,dtype=bool),.9,0.,np.zeros(0),np.zeros((3,0)))
 r=[];c=[];d=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for a,b,e in pool.map(run,[4,10,13,14,15,16,18,19,20]):
   r+=a;c+=b;d+=e;print('DONE',a[0]['zone'],flush=True)
 for data,name in [(r,'all_scores_MW.csv'),(c,'forecast_verification.csv'),(d,'objective_verification.csv')]:pd.DataFrame(data).to_csv(O/name,index=False)
