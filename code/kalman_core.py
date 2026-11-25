"""Gaussian innovation gradient and filtered state recursions."""
import numpy as np
from numba import njit

@njit(cache=True)
def gradient(r,mask,rho,a,g,Z,ss,nobs):
 k=2+len(g);s=0.;P=1.;ds=np.zeros(k);dP=np.zeros(k);dss=np.zeros(k);dld=np.zeros(k);q=1-rho*rho
 for t in range(len(r)):
  if mask[t]:
   zz=0.
   for j in range(len(g)):zz+=Z[t,j]*g[j]
   ex=np.exp(zz);nu=a*ex;V=P+nu;K=P/V;e=r[t]-s
   for j in range(k):
    dn=0. if j==0 else (ex if j==1 else nu*Z[t,j-2]);dv=dP[j]+dn;dk=(dP[j]-K*dv)/V
    dss[j]+=-2*e*ds[j]/V-e*e*dv/(V*V);dld[j]+=dv/V;ds[j]=(1-K)*ds[j]+e*dk;dP[j]=(1-K)*dP[j]-P*dk
   s+=K*e;P*=1-K
  ds*=rho;ds[0]+=s;s*=rho;dP*=rho*rho;dP[0]+=2*rho*(P-1);P=rho*rho*P+q
 return nobs/ss*dss+dld


@njit(cache=True)
def states(W,mask,rho,a,g,Z):
 n,m=W.shape;s=np.zeros(m);out=np.empty((n,m));pv=np.empty(n);P=1.;q=1-rho*rho
 for t in range(n):
  if mask[t]:
   zz=0.
   for j in range(len(g)):zz+=Z[t,j]*g[j]
   K=P/(P+a*np.exp(zz));s+=K*(W[t]-s);P*=1-K
  out[t]=s;pv[t]=P;s*=rho;P=rho*rho*P+q
 return out,pv

