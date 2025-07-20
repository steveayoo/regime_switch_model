import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, laplace

# -----------------------------
# Step 1: Load Excel Files
# -----------------------------
bunker_df = pd.read_excel('data/Bunker prices.xlsx', skiprows=5)
bulker_df = pd.read_excel('data/Bulker prices.xlsx', skiprows=5)

# Drop unnamed columns and convert dates
bunker_df = bunker_df.loc[:, ~bunker_df.columns.str.contains('^Unnamed')]
bulker_df = bulker_df.loc[:, ~bulker_df.columns.str.contains('^Unnamed')]
bunker_df['Date'] = pd.to_datetime(bunker_df['Date'])
bulker_df['Date'] = pd.to_datetime(bulker_df['Date'])

# Rename bunker_df columns for clarity
bunker_df.columns = [
    'Date',
    'HSFO_SG','VLSFO_SG','Spread_SG',
    'VLSFO_Rot','HSFO_Rot','Spread_Rot',
    'HSFO_Fuj','VLSFO_Fuj','Spread_Fuj',
    'HSFO_Hou','VLSFO_Hou','Spread_Hou',
    'HSFO_Pan','VLSFO_Pan','Spread_Pan'
]

# Merge on Date
df = pd.merge(bunker_df, bulker_df, on='Date', how='left')

# -----------------------------
# Step 2: Identify Regimes with GMM
# -----------------------------
regions = ['SG', 'Rot', 'Fuj', 'Hou', 'Pan']
regime_labels = pd.DataFrame(df[['Date']])

for hub in regions:
    spread = df[['Date', f'Spread_{hub}']].dropna().copy()
    spread['Change'] = spread[f'Spread_{hub}'].diff()
    spread.dropna(inplace=True)

    gmm = GaussianMixture(n_components=2, random_state=42).fit(spread[['Change']])
    labels = gmm.predict(spread[['Change']])
    variances = gmm.covariances_.flatten()
    jump_label = np.argmax(variances)

    spread['Regime'] = np.where(labels == jump_label, 'Jump', 'Base')
    regime_labels = pd.merge(regime_labels, spread[['Date', 'Regime']], on='Date', how='left')
    regime_labels.rename(columns={'Regime': f'Regime_{hub}'}, inplace=True)

regime_labels.to_csv('regime_labels.csv', index=False)

# -----------------------------
# Step 3: Estimate RSJD Parameters
# -----------------------------
param_results = []

for hub in regions:
    series = df[['Date', f'Spread_{hub}']].dropna().copy()
    series['Change'] = series[f'Spread_{hub}'].diff()
    series.dropna(inplace=True)
    series = pd.merge(series, regime_labels[['Date', f'Regime_{hub}']], on='Date', how='left')

    for regime in ['Base', 'Jump']:
        subset = series[series[f'Regime_{hub}'] == regime]['Change']
        if len(subset) == 0:
            continue
        drift = subset.mean()
        volatility = subset.std()
        lam = 0.0
        if regime == 'Jump':
            lam = (len(subset) / len(series)) * 52
        param_results.append({
            'Region': hub,
            'Regime': regime,
            'Drift_per_week': round(drift, 4),
            'Volatility_per_week': round(volatility, 4),
            'Jump_Intensity_per_year': round(lam, 2)
        })

param_df = pd.DataFrame(param_results)
param_df.to_csv('RSJD_parameters.csv', index=False)

# -----------------------------
# Step 4: Jump Size Distribution Diagnostics
# -----------------------------
comparison = []

for hub in regions:
    merged = pd.merge(df[['Date', f'Spread_{hub}']], regime_labels[['Date', f'Regime_{hub}']], on='Date', how='left')
    merged['Change'] = merged[f'Spread_{hub}'].diff()
    jump_data = merged[merged[f'Regime_{hub}'] == 'Jump']['Change'].dropna()

    if len(jump_data) < 5:
        continue

    # Fit normal
    mu, sigma = norm.fit(jump_data)
    loglik_norm = np.sum(norm.logpdf(jump_data, mu, sigma))
    aic_norm = 2*2 - 2*loglik_norm
    bic_norm = np.log(len(jump_data))*2 - 2*loglik_norm

    # Fit Laplace
    loc, scale = laplace.fit(jump_data)
    loglik_lap = np.sum(laplace.logpdf(jump_data, loc, scale))
    aic_lap = 2*2 - 2*loglik_lap
    bic_lap = np.log(len(jump_data))*2 - 2*loglik_lap

    comparison.append({
        'Region': hub,
        'Normal_AIC': round(aic_norm, 2),
        'Normal_BIC': round(bic_norm, 2),
        'Laplace_AIC': round(aic_lap, 2),
        'Laplace_BIC': round(bic_lap, 2)
    })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv('jump_distribution_comparison.csv', index=False)

# -----------------------------
# Final Message
# -----------------------------
print("RSJD model calibration completed successfully.")
print("Outputs saved:")
print(" - regime_labels.csv")
print(" - RSJD_parameters.csv")
print(" - jump_distribution_comparison.csv")
