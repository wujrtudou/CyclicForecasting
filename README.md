# GEFCom2012: Multi-step forecasting with cyclic heterogeneous nuggets

Replication materials for the electricity-demand application of sparse precision methods for correlated time series with heterogeneous observation errors.

## Experimental design

- Zones: 4, 10, 13, 14, 15, 16, 18, 19 and 20.
- Training: 2005–2006; evaluation: 2007.
- Forecast horizons: 1, 3, 6 and 12 hours; 1,433 origins per zone and horizon.
- Fixed 285-column calendar and cubic-temperature regression design.
- Models: M0 (no nugget), M1 (homogeneous nugget), M2d (daily cyclic nugget), M2dw (daily and weekly cyclic nugget), all with latent AR(1) dependence.
- Contemporaneous observed temperature is used for conditional forecasting. No artificial contamination is introduced.

The nine zones were selected retrospectively from 20 zones; results are exploratory and do not represent an unbiased estimate of performance across all zones. Some Fourier coefficients reach their optimization bounds.

## Repository contents

| Path | Description |
| --- | --- |
| `code/fit_selected9.py` | Profile likelihood fitting and forecasting |
| `code/pairwise_core.py` | Data preprocessing, station selection and fixed regression design |
| `code/kalman_core.py` | Gaussian innovation gradient and state recursions |
| `code/warm_start_parameters.csv` | Historical optimization starting values |
| `code/replay_sparse.py` | Fixed-parameter sparse GLS recomputation and state updates |
| `code/verify_scores.py` | Independent reconstruction of archived scores |
| `data/raw_inputs.zip` | Original load and temperature CSV files |
| `results/scores_MW.csv` | 144 score rows, including RMSE, MAE, CRPS, coverage and interval score |
| `results/fitted_parameters.csv` | 36 fitted covariance parameter sets |
| `results/*_table.csv` | RMSE, MAE and CRPS tables in MW |
| `results/forecasts/` | 206,352 individual forecasts in gzip-compressed CSV format |
| `results/verification/` | Score reconstruction and Zone 10 numerical and paired CRPS checks |
| `paper/` | LaTeX section, supplementary tables and bibliography |

## Reproduction

Python 3.11 or later is recommended. From the repository root:

```sh
python -m pip install -r requirements.txt
python code/verify_scores.py
```

Refit all nine zones:

```sh
python code/fit_selected9.py --out refit_output --workers 2
```

For a single zone:

```sh
python code/fit_selected9.py --out zone10_output --zones 10 --workers 1
```

The fitting driver retains the original eight-horizon eligibility rule for forecast origins to match the archived sample. Numerical optimization may vary across platforms and libraries. For fixed-parameter sparse recomputation:

```sh
python code/replay_sparse.py
```

The replay writes to `replay_output/`. It does not re-optimize covariance parameters. Archived results are not overwritten. The score verification checks all 144 rows against individual predictions; this is an aggregation check, not independent optimizer replication. Archived sparse numerical checks and paired CRPS confidence intervals concern Zone 10 only; the intervals are exploratory and unadjusted for multiple comparisons.

## Data and references

The load and temperature histories originate from the Global Energy Forecasting Competition 2012 (GEFCom2012). See Hong, Pinson and Fan (2014), *International Journal of Forecasting*, 30(2), 357–363, https://doi.org/10.1016/j.ijforecast.2013.07.001. The mean design follows the calendar–temperature polynomial benchmark tradition. Further references are provided in `paper/gefcom_hourly_references.bib`.

Third-party data remain subject to their original terms of use. No new data redistribution license is asserted by this repository.
