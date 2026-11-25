"""Fit the four covariance models using the fixed 285-column mean design."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys, argparse, zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import cho_factor,cho_solve
from scipy.special import ndtr
from numba import njit
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from concurrent.futures import ProcessPoolExecutor,as_completed
from kalman_core import gradient,states
H=np.array([1,3,6,12,24,48,72,168]);MODELS=['M0','M1','M2d','M2dw'];N0=17520
@njit(cache=True)
def white_compact(W,gaps,rho,a,g,Z):
 n,m=W.shape;w=np.empty((n,m));s=np.zeros(m);P=1.;ld=0.
 for t in range(n):
  ph=rho**gaps[t];P=1-ph*ph*(1-P);s*=ph
  ex=0.
  for j in range(len(g)):ex+=Z[t,j]*g[j]
  nu=a*np.exp(ex);v=P+nu;k=P/v;rt=np.sqrt(v);ld+=np.log(v)
  for j in range(m):
   e=W[t,j]-s[j];w[t,j]=e/rt;s[j]+=k*e
  P*=1-k
 return w,ld
@njit(cache=True)
def residual_obj(r,mask,rho,a,g,Z):
 s=0.;P=1.;ld=0.;ss=0.;q=1-rho*rho
 for t in range(len(r)):
  if mask[t]:
   ex=0.
   for j in range(len(g)):ex+=Z[t,j]*g[j]
   V=P+a*np.exp(ex);K=P/V;e=r[t]-s;ss+=e*e/V;ld+=np.log(V);s+=K*e;P*=1-K
  s*=rho;P=rho*rho*P+q
 return ss,ld

def fit(W,mask,Z,dim,previous,initial=None,maxiter=15):
 n=int(mask.sum());wc=np.ascontiguousarray(W[mask]);ix=np.flatnonzero(mask);gaps=np.diff(ix,prepend=ix[0]);zm=Z[mask,:dim].mean(0);zz=np.ascontiguousarray(Z[:,:dim]-zm);zc=np.ascontiguousarray(zz[mask]);X=W[:,1:];y=W[:,0]
 bounds=[(.01001,.9999),(0.,0. if previous is None else 5.)]+[(-2.5,2.5)]*dim
 calls=0
 def raw(p,full=False):
  nonlocal calls
  calls+=1;w,ld=white_compact(wc,gaps,p[0],p[1],p[2:],zc);C=w.T@w;G=C[1:,1:];b=cho_solve(cho_factor(G,check_finite=False),C[1:,0],check_finite=False);e=w[:,0]-w[:,1:]@b;ss=e@e;val=n*np.log(ss/n)+ld
  if full:return val,b,ss/n,G
  return val/n,gradient(y-X@b,mask,p[0],p[1],p[2:],zz,ss,n)/n
 def fixedfit(b,p):
  r=y-X@b
  def f(q):
   ss,ld=residual_obj(r,mask,q[0],q[1],q[2:],zz)
   return np.log(ss/n)+ld/n,gradient(r,mask,q[0],q[1],q[2:],zz,ss,n)/n
  return minimize(f,p,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':80,'ftol':1e-11,'gtol':1e-7}).x
 def projected(p,gr):
  proj=gr.copy()
  for j,(lo,hi) in enumerate(bounds):
   if (p[j]<=lo+1e-8 and gr[j]>0) or (p[j]>=hi-1e-8 and gr[j]<0):proj[j]=0.
  return float(np.max(abs(proj)))
 if previous is None:
  # Exact AR(1) no-nugget Gram from time-gap sufficient statistics.
  products=[]
  for d in np.unique(gaps[1:]):
   ids=np.flatnonzero(gaps==d);A=wc[ids];B=wc[ids-1];AB=A.T@B;products.append((d,len(ids),A.T@A,B.T@B,AB+AB.T))
  c0=np.outer(wc[0],wc[0]);
  def m0(rho):
   C=c0.copy();ld=0.
   for d,num,A,B,AB in products:
    ph=rho**d;q=1-ph*ph;C+=(A+ph*ph*B-ph*AB)/q;ld+=num*np.log(q)
   G=C[1:,1:];b=cho_solve(cho_factor(G,check_finite=False),C[1:,0],check_finite=False);ss=C[0,0]-C[0,1:]@b
   return np.log(ss/n)+ld/n
  sol=minimize_scalar(m0,bounds=(.01001,.9999),method='bounded',options={'xatol':1e-10});best=np.array([sol.x,0.]);msg=str(sol.message);ok=sol.success
 else:
  base=np.pad(previous['p'],(0,2+dim-len(previous['p'])));b=previous['b'];starts=[base]
  # Fit the cheap scalar-residual model from multiple seeds, then compare
  # exact profile values. Refit GLS and covariance alternately, refine if needed.
  for seed in ([.9,.02]+[.2]*dim,[.98,.015]+([-1.5,-.6]+[0.]*(dim-2) if dim else [])):
   starts.append(fixedfit(b,np.array(seed)))
  if initial is not None and np.all(np.isfinite(initial)):starts.append(initial)
  # Deduplicate candidates at the boundary (gamma does not matter at a=0).
  unique=[]
  for p in starts:
   if p[1]<1e-9:p=p.copy();p[1:]=0.
   if not any(np.max(abs(p-q))<1e-5 for q in unique):unique.append(p)
  cand=[(raw(p)[0],p) for p in unique];cand.sort(key=lambda x:x[0]);bestv,best=cand[0]
  val,b,scale,G=raw(best,True)
  # Two conditional refinements provide inexpensive warm starts.
  for it in range(2):
   q=fixedfit(b,best)
   if np.max(abs(q-best))<1e-6:break
   vv,bb,ss,GG=raw(q,True)
   if vv/n>=bestv-1e-10:break
   bestv,best,b=vv/n,q,bb
  _,gr=raw(best);pg=projected(best,gr)
  if pg<1e-5:ok=True;msg='Projected score tolerance reached'
  else:
   if best[1]>1e-8:
    def logobj(q):
     p=q.copy();p[1]=np.exp(q[1]);v,g=raw(p);g[1]*=p[1];return v,g
    q=best.copy();q[1]=np.log(q[1]);bb=[bounds[0],(np.log(1e-10),np.log(5.))]+bounds[2:];sol=minimize(logobj,q,jac=True,bounds=bb,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-7,'maxls':20});pp=sol.x.copy();pp[1]=np.exp(pp[1])
   else:
    sol=minimize(raw,best,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':15});pp=sol.x
   ok=sol.success;msg=str(sol.message)
   if sol.fun<=bestv+1e-10:best=pp
 value,b,scale,G=raw(best,True);_,gr=raw(best);pg=projected(best,gr)
 return dict(p=best,b=b,scale=scale,G=G,zm=zm,obj=value,ok=bool(ok or pg<1e-5),message=msg,calls=calls,projected_gradient=pg)

def run(zone,mode,datadir,out,maxiter,test_year=2007,baseline_only=False):
 sys.path.insert(0,str(datadir));from pairwise_core import get_data,choose_station,make_table,build_design
 zones,temps=get_data(datadir,[zone]);df=zones[zone]
 if mode=='lag168':temps=temps.copy();temps.index+=pd.Timedelta(hours=168)
 station=choose_station(df,temps,'auto') # frozen 2005-06 selection for both evaluations
 if test_year==2007:table=make_table(df,temps,station,trend_rule='row' if mode=='actual' else 'elapsed')
 else:
  table=df.join(temps[[station]],how='left').rename(columns={station:'T'}).replace([np.inf,-np.inf],np.nan).dropna(subset=['load_mw','T'])
  table['trend_hours']=np.arange(1,len(table)+1,dtype=float) if mode=='actual' else ((table.index-pd.Timestamp('2004-01-01'))/pd.Timedelta(hours=1))+1
  table=table[(table.index.year>=2005)&(table.index.year<=test_year)].copy();table['month']=table.index.month;table['weekday']=table.index.dayofweek;table['hour']=table.index.hour
 train=table[table.index.year<test_year];test=table[table.index.year==test_year];X,V,names,_=build_design(train,test)
 dt=pd.date_range('2005-01-01','2007-12-31 23:00' if test_year==2007 else '2008-06-30 23:00',freq='h');n=len(dt);N0=int(dt.searchsorted(pd.Timestamp(f'{test_year}-01-01')));loc=dt.get_indexer(table.index);assert (loc>=0).all();mask=np.zeros(n,bool);mask[loc]=True;W=np.zeros((n,286));W[loc,0]=table.load_mw.to_numpy();W[loc,1:]=np.vstack([X,V]);tt=np.arange(n);Z=np.column_stack([np.sin(2*np.pi*tt/24),np.cos(2*np.pi*tt/24),np.sin(2*np.pi*tt/168),np.cos(2*np.pi*tt/168)])
 lags=np.array([1,2,3,6,12,24,25,48,72,168]);orig=np.arange(N0,n-167,6);orig=orig[mask[orig[:,None]-lags].all(axis=1)&mask[orig[:,None]+H-1].all(axis=1)&mask[orig[:,None]+H-1-168].all(axis=1)];rows=[];fits=[];details=[];nested=None;start=time.time()
 def score(mod,h,mu,var):
  actual=W[orig+h-1,0];sd=np.sqrt(np.maximum(var,1e-12));e=actual-mu;z=e/sd;crps=sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi));width=1.6448536269514722*sd
  rows.append(dict(zone=zone,station=station,mode=mode,model=mod,h=h,n=len(e),RMSE_MW=np.sqrt(np.mean(e*e)),MAE_MW=np.mean(np.abs(e)),CRPS_MW=crps.mean(),coverage90=np.mean(np.abs(e)<=width),interval_score90_MW=np.mean(2*width+20*np.maximum(np.abs(e)-width,0)),train_mean_MW=train.load_mw.mean()))
  details.append(pd.DataFrame(dict(origin=dt[orig-1],target=dt[orig+h-1],h=h,model=mod,actual_MW=actual,forecast_MW=mu,pred_sd_MW=sd,CRPS_MW=crps)))
 pilot=Path(__file__).with_name('warm_start_parameters.csv');prior=pd.read_csv(pilot) if mode=='actual' and test_year==2007 and pilot.exists() else None
 if prior is not None:prior=prior[prior.zone==zone];prior=prior if len(prior)==4 else None
 for model,dim in zip(MODELS,[0,0,2,4]):
  init=None
  if prior is not None:
   pr=prior[prior.model==model];init=pr[['rho','a']+[f'gamma{i+1}' for i in range(dim)]].to_numpy()[0]
  f=fit(W[:N0],mask[:N0],Z[:N0],dim,nested,init,maxiter);nested=f;p=f['p'];rho,a=p[:2];g=p[2:];b=f['b'];z=Z[:,:dim]-f['zm'];s=f['scale'];G=f['G'];st,P=states(W,mask,rho,a,g,np.ascontiguousarray(z));st=st[orig-1];state=st[:,0]-st[:,1:]@b;post=P[orig-1];cf=cho_factor(G,check_finite=False);ratio=a*np.exp(z[:N0][mask[:N0]]@g)
  fits.append(dict(zone=zone,station=station,mode=mode,model=model,n_train=len(train),n_test=len(test),rho=rho,a=a,success=f['ok'],objective=f['obj'],scale_MW2=s,mean_nugget_fraction=np.mean(ratio/(1+ratio)),message=f['message'],calls=f['calls'],projected_gradient=f['projected_gradient'],**{f'gamma{i+1}':x for i,x in enumerate(g)}))
  for h in [1,3,6,12]:
   tar=orig+h-1;fX=W[tar,1:];phi=rho**h;mu=fX@b+phi*state;A=fX-phi*st[:,1:];var=s*(1-phi*phi*(1-post)+a*np.exp(z[tar]@g)+np.sum(A*cho_solve(cf,A.T,check_finite=False).T,axis=1));score(model,h,mu,var)
  print(mode,zone,model,'rho',round(rho,5),'a',round(a,6),'ok',f['ok'],'calls',f['calls'],'sec',round(time.time()-start,1),flush=True)
 pd.DataFrame(rows).to_csv(out/f'scores_{mode}_{zone:02d}.csv',index=False);pd.DataFrame(fits).to_csv(out/f'params_{mode}_{zone:02d}.csv',index=False)
 pd.concat(details,ignore_index=True).to_csv(out/f'preds_{mode}_{zone:02d}.csv.gz',index=False,compression='gzip');return mode,zone
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--zip',default=str(Path(__file__).resolve().parents[1]/'data/raw_inputs.zip'));pa.add_argument('--out',required=True);pa.add_argument('--workers',type=int,default=4);pa.add_argument('--modes',default='actual');pa.add_argument('--zones',default='4,10,13,14,15,16,18,19,20');pa.add_argument('--maxiter',type=int,default=25);pa.add_argument('--test-year',type=int,choices=[2007,2008],default=2007);pa.add_argument('--baseline-only',action='store_true');a=pa.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);d=out/'input';d.mkdir(exist_ok=True)
 with zipfile.ZipFile(a.zip) as z:
  for name in ['Load_history.csv','temperature_history.csv']:z.extract(name,d)
 w=np.ones((12,2));m=np.ones(12,dtype=bool);z=np.zeros((12,0));g=np.zeros(0);white_compact(w,np.ones(12,dtype=np.int64),.9,.1,g,z);residual_obj(w[:,0],m,.9,.1,g,z);gradient(w[:,0],m,.9,.1,g,z,1.,12);states(w,m,.9,.1,g,z)
 zones=list(range(1,21)) if a.zones=='all' else [int(x) for x in a.zones.split(',')];tasks=[(z,m) for m in a.modes.split(',') for z in zones];t=time.time()
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(run,z,m,d,out,a.maxiter,a.test_year,a.baseline_only) for z,m in tasks if a.baseline_only or not (out/f'preds_{m}_{z:02d}.csv.gz').exists()]
  for f in as_completed(futures):
   try:print('DONE',f.result(),round(time.time()-t,1),flush=True)
   except Exception as e:print('FAILED',repr(e),flush=True)
