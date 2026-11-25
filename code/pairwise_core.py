"""GEFCom2012 data preprocessing and fixed 285-column regression design."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd


def locate_csv(datadir: Path, name: str) -> Path:
    for path in (datadir/name, datadir/'raw'/name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name}; put the original CSV in {datadir}")


def read_wide_csv(path: Path, idtype: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = ['year', 'month', 'day']
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError(f'{path.name}: missing calendar columns: {missing}; found {raw.columns.tolist()[:8]}')
    hcols = [c for c in raw if re.fullmatch(r'h(?:0?[1-9]|1[0-9]|2[0-4])', c)]
    if len(hcols) != 24:
        # Files sometimes use 1, 2, ..., 24 as column names.
        hcols = [c for c in raw if re.fullmatch(r'(?:[1-9]|1[0-9]|2[0-4])', c)]
    if len(hcols) != 24:
        raise ValueError(f'{path.name}: expected 24 hourly columns, got {len(hcols)}')
    id_col = next((c for c in raw if idtype in c), None)
    if id_col is None:
        id_col = next((c for c in raw if c not in required+hcols), None)
    if id_col is None:
        raise ValueError(f'{path.name}: cannot identify zone/station column')
    raw = raw.rename(columns={id_col: 'id'})
    for c in ['id'] + required:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    return raw, hcols


def melt_hourly(wide: pd.DataFrame, hcols: list[str], name: str) -> pd.DataFrame:
    long = wide.melt(id_vars=['id','year','month','day'], value_vars=hcols,
                     var_name='hour_col', value_name=name)
    long['hour_0to23'] = long['hour_col'].str.extract(r'(\d+)').astype(int)-1
    long['date'] = pd.to_datetime(long[['year','month','day']], errors='coerce')
    long['datetime'] = long['date'] + pd.to_timedelta(long['hour_0to23'], unit='h')
    # CSV often uses quoted numbers such as "16,853".
    if not pd.api.types.is_numeric_dtype(long[name]):
        long[name] = pd.to_numeric(long[name].astype(str).str.replace(',','',regex=False), errors='coerce')
    long = long[['id','datetime',name]].dropna(subset=['id','datetime'])
    long['id'] = long['id'].astype(int)
    return long


def get_data(datadir: Path, zone_ids: list[int]) -> tuple[dict[int,pd.DataFrame], pd.DataFrame]:
    lraw, lhours = read_wide_csv(locate_csv(datadir, 'Load_history.csv'), 'zone')
    lraw = lraw[lraw['id'].isin(zone_ids)].copy()
    if lraw.empty: raise ValueError(f'None of zones {zone_ids} found in load file')
    load = melt_hourly(lraw, lhours, 'load_kw')
    load['load_mw'] = load['load_kw']/1000.
    traw, thours = read_wide_csv(locate_csv(datadir, 'temperature_history.csv'), 'station')
    temp = melt_hourly(traw, thours, 'temperature')
    temp = temp.pivot_table(index='datetime', columns='id', values='temperature', aggfunc='first')
    temp = temp.sort_index()
    temp.columns = [f'station_{s}' for s in temp.columns]
    zones = {}
    for zone in zone_ids:
        sub = load[load['id']==zone].drop(columns=['id','load_kw'])
        sub = sub.set_index('datetime').sort_index()
        if not sub.index.is_unique: raise ValueError(f'duplicate timestamps in zone {zone}')
        zones[zone] = sub
    return zones,temp


def choose_station(zone_df: pd.DataFrame, temps: pd.DataFrame, rule: str) -> str:
    if rule != 'auto':
        key = f'station_{int(rule)}'
        if key not in temps: raise ValueError(f'Temperature {key} not available')
        return key
    # Station selection based on 2005--06 training rows only; 2007 is sealed.
    df = zone_df.join(temps, how='left')
    df = df[(df.index.year>=2005)&(df.index.year<=2006)]
    best_key,best=-1,-np.inf
    for c in temps.columns:
        arr=df[['load_mw',c]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(arr)<100: continue
        corr=abs(arr['load_mw'].corr(arr[c]))
        if np.isfinite(corr) and corr>best:
            best_key,best=c,corr
    if best_key==-1: raise ValueError('no valid temperature station during training')
    return best_key


def make_table(zone_df: pd.DataFrame, temps: pd.DataFrame, station: str,
               trend_rule: str='row') -> pd.DataFrame:
    df=zone_df.join(temps[[station]], how='left').rename(columns={station:'T'})
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['load_mw','T'])
    # The Hong benchmark describes L as the row index of the cleaned data.
    # Count from the full 2004 history before selecting training/testing years.
    # 'elapsed' preserves physical-hour distances across missing intervals and
    # is available as a sensitivity analysis, but is NOT the paper default.
    if trend_rule=='row':
        df['trend_hours']=np.arange(1,len(df)+1,dtype=float)
    elif trend_rule=='elapsed':
        df['trend_hours']=((df.index-pd.Timestamp('2004-01-01'))/pd.Timedelta(hours=1))+1
    else:
        raise ValueError('trend_rule must be row or elapsed')
    df=df[(df.index.year>=2005)&(df.index.year<=2007)].copy()
    df['month']=df.index.month
    df['weekday']=df.index.dayofweek       # Monday=0 .. Sunday=6
    df['hour']=df.index.hour
    return df


def build_design(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,list[str],dict]:
    """Full-rank reference coding: 1+1+11+6+23+3+138+33+69 = 285."""
    tm=float(train['T'].mean()); ts=float(train['T'].std(ddof=0))
    lm=float(train['trend_hours'].mean()); ls=float(train['trend_hours'].std(ddof=0))
    if ts<1e-6 or ls<1e-6: raise ValueError('insufficient train temperature or trend variation')
    def design(df):
        n=len(df); T=((df['T'].to_numpy()-tm)/ts)
        L=(df['trend_hours'].to_numpy()-lm)/ls
        M=(df['month'].to_numpy()[:,None] == np.arange(2,13)[None,:]).astype(float)
        W=(df['weekday'].to_numpy()[:,None] == np.arange(1,7)[None,:]).astype(float)
        H=(df['hour'].to_numpy()[:,None] == np.arange(1,24)[None,:]).astype(float)
        poly=np.column_stack([T,T*T,T*T*T])
        HW=(H[:,:,None]*W[:,None,:]).reshape(n,-1)
        TM=(poly[:,:,None]*M[:,None,:]).reshape(n,-1)
        TH=(poly[:,:,None]*H[:,None,:]).reshape(n,-1)
        return np.column_stack([np.ones(n),L,M,W,H,poly,HW,TM,TH])
    X=design(train); V=design(valid)
    names=(['intercept','trend']+[f'month_{m}' for m in range(2,13)]+
       [f'weekday_{w}' for w in range(1,7)]+[f'hour_{h}' for h in range(1,24)]+
       [f'T^{j}' for j in (1,2,3)]+
       [f'hour_{h}:weekday_{w}' for h in range(1,24) for w in range(1,7)]+
       [f'T^{j}:month_{m}' for j in (1,2,3) for m in range(2,13)]+
       [f'T^{j}:hour_{h}' for j in (1,2,3) for h in range(1,24)])
    assert X.shape[1]==285 and V.shape[1]==285 and len(names)==285
    # Changing regressor units does not change the fitted model; stabilizes Newton.
    center=X[:,1:].mean(axis=0)
    spread=X[:,1:].std(axis=0)
    spread=np.maximum(spread,1e-9)
    X[:,1:] = (X[:,1:]-center)/spread
    V[:,1:] = (V[:,1:]-center)/spread
    return X,V,names,{'temp_mean':tm,'temp_sd':ts,'trend_mean':lm,'trend_sd':ls,
                     'column_center':center.tolist(),'column_sd':spread.tolist()}
