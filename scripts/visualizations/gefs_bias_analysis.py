"""
GEFS Ensemble Precipitation Forecast Bias Analysis
====================================================
Investigates whether GEFS ensemble mean precipitation forecasts
systematically over/under-predict compared to observed precipitation.

Analyzes the training dataset across horizons, seasons, and sources.

Usage: python gefs_bias_analysis.py
"""

import pandas as pd
import numpy as np
import json
import os
import pickle

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training', 'training_dataset.parquet')
PARAMS_35D = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgboostlss_35d_params.json')
PARAMS_7D = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgboostlss_7d_params.json')
MODEL_35D = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgboostlss_35d.json')

df = pd.read_parquet(DATA_PATH)
print(f'Loaded {len(df):,} rows, {len(df.columns)} columns')
print(f'Horizon values: {sorted(df["horizon_days"].unique())}')
print(f'Source values: {df["source"].unique()}')
print(f'Date range: {df["init_date"].min()} to {df["init_date"].max()}')
print()

# =================================================================
# PART 1: 35-day horizon overall bias
# =================================================================
print('=' * 72)
print('PART 1: 35-DAY HORIZON -- GEFS vs OBSERVED PRECIPITATION')
print('=' * 72)

h35 = df[df['horizon_days'] == 35].copy()
print(f'N rows (35d): {len(h35):,}')

forecast = h35['cum_apcp_mean']
observed = h35['ground_truth_precip']
bias = forecast - observed

for label, val_f, val_o in [
    ('Mean', forecast.mean(), observed.mean()),
    ('Median', forecast.median(), observed.median()),
    ('Std Dev', forecast.std(), observed.std()),
    ('25th pctile', forecast.quantile(0.25), observed.quantile(0.25)),
    ('75th pctile', forecast.quantile(0.75), observed.quantile(0.75)),
]:
    print(f'{label+" (mm)":<35} Fcst={val_f:>8.1f}  Obs={val_o:>8.1f}  Diff={val_f-val_o:>+8.1f}')

print(f'Pearson r = {forecast.corr(observed):.4f}')
print(f'Spearman r = {forecast.rank().corr(observed.rank()):.4f}')
print(f'Over-predicts: {(forecast > observed).mean()*100:.1f}%')
print(f'Under-predicts: {(forecast < observed).mean()*100:.1f}%')
print(f'MAE = {bias.abs().mean():.1f} mm')
print(f'RMSE = {np.sqrt((bias**2).mean()):.1f} mm')

# =================================================================
# PART 2: Season and source breakdown
# =================================================================
print()
print('=' * 72)
print('PART 2: BREAKDOWN BY SEASON AND SOURCE (35d)')
print('=' * 72)

h35['season'] = np.where(h35['is_wet_season']==1, 'Wet (Jun-Oct)', 'Dry (Nov-May)')
h35['bias'] = h35['cum_apcp_mean'] - h35['ground_truth_precip']
h35['over_pred'] = (h35['cum_apcp_mean'] > h35['ground_truth_precip']).astype(int)

for groupcol in ['season', 'source']:
    print(f'--- By {groupcol} ---')
    for val, grp in h35.groupby(groupcol):
        n = len(grp)
        mb = grp['bias'].mean()
        op = grp['over_pred'].mean()*100
        c = grp['cum_apcp_mean'].corr(grp['ground_truth_precip'])
        print(f'  {val:<20} N={n:>5} bias={mb:>+7.1f}mm over-pred={op:.1f}% corr={c:.3f}')

# =================================================================
# PART 3: Non-linear bias by range
# =================================================================
print()
print('=' * 72)
print('PART 3: NON-LINEAR BIAS BY OBSERVED RANGE (35d)')
print('=' * 72)

bins = [0, 20, 50, 100, 150, 200, 300, 500, 800]
labels = ['0-20','20-50','50-100','100-150','150-200','200-300','300-500','500+']
h35['obs_bin'] = pd.cut(h35['ground_truth_precip'], bins=bins, labels=labels)
for label in labels:
    grp = h35[h35['obs_bin']==label]
    if len(grp)==0: continue
    mb = grp['bias'].mean()
    mo = grp['ground_truth_precip'].mean()
    rb = mb/mo*100 if mo>0 else 0
    op = grp['over_pred'].mean()*100
    print(f'  Obs {label:<10} N={len(grp):>5} bias={mb:>+8.1f}mm rel={rb:>+7.1f}% over-pred={op:.0f}%')

# =================================================================
# PART 4: All horizons
# =================================================================
print()
print('=' * 72)
print('PART 4: BIAS ACROSS ALL HORIZONS')
print('=' * 72)

for hz in sorted(df['horizon_days'].unique()):
    hdf = df[df['horizon_days']==hz]
    fc = hdf['cum_apcp_mean']; ob = hdf['ground_truth_precip']
    b = fc - ob
    print(f'  {hz:>2}d: N={len(hdf):>5} fcst={fc.mean():>7.1f} obs={ob.mean():>7.1f} bias={b.mean():>+7.1f} over-pred={((fc>ob).mean()*100):.0f}% corr={fc.corr(ob):.3f} MAE={b.abs().mean():.1f}')

# =================================================================
# PART 5: Feature importance
# =================================================================
print()
print('=' * 72)
print('PART 5: MODEL FEATURE IMPORTANCE')
print('=' * 72)

if os.path.exists(MODEL_35D):
    with open(MODEL_35D, 'rb') as f:
        model_obj = pickle.load(f)
    booster = model_obj.booster
    importance = booster.get_score(importance_type='gain')
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print('Top 20 features by gain:')
    for i, (feat, gain) in enumerate(imp_sorted[:20], 1):
        print(f'  {i:>3}. {feat:<40} gain={gain:.1f}')

print()
print('Done.')
