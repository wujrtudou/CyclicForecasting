from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr

root = Path(__file__).resolve().parents[1]
ref = pd.read_csv(root/'results/scores_MW.csv').set_index(['zone','h','model'])
rows = []
for file in sorted((root/'results/forecasts').glob('zone_*.csv.gz')):
    d = pd.read_csv(file)
    for key, g in d.groupby(['zone','h','model']):
        e = g.actual_MW-g.forecast_MW
        sd = g.pred_sd_MW
        z = e/sd
        crps = sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
        w = 1.6448536269514722*sd
        calc = dict(n=len(g), RMSE_MW=np.sqrt(np.mean(e**2)), MAE_MW=abs(e).mean(), CRPS_MW=crps.mean(), coverage90=(abs(e)<=w).mean(), interval_score90_MW=(2*w+20*np.maximum(abs(e)-w,0)).mean())
        rows.append(dict(zone=key[0], h=key[1], model=key[2], **{k:abs(v-ref.loc[key,k]) for k,v in calc.items()}))
check = pd.DataFrame(rows)
assert len(check)==len(ref)==144
assert check.iloc[:,3:].to_numpy().max()<1e-8
print('Verified 144 score rows. Maximum absolute discrepancies:')
print(check.iloc[:,3:].max().to_string())
