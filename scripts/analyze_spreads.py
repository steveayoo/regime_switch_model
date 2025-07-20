import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera, norm

# ==================================
# DATA IMPORT AND PREPARATION
# ==================================
file_path = r'C:\Users\HomePC\OneDrive\Desktop\bunker_fuel_analysis\data\Bunker prices.xlsx'

# Read data with proper column specification
df = pd.read_excel(
    file_path,
    skiprows=5,
    usecols="K:Z",
    parse_dates=['Date']
)

# Set column names correctly
df.columns = [
    'Date',
    'SG_HSFO', 'SG_VLSFO', 'SG_Spread',
    'Rot_VLSFO', 'Rot_HSFO', 'Rot_Spread',
    'Fuj_HSFO', 'Fuj_VLSFO', 'Fuj_Spread',
    'Hou_HSFO', 'Hou_VLSFO', 'Hou_Spread',
    'Pan_HSFO', 'Pan_VLSFO', 'Pan_Spread'
]

# Verify spreads
df['SG_Spread_calc'] = df['SG_VLSFO'] - df['SG_HSFO']
df['Rot_Spread_calc'] = df['Rot_VLSFO'] - df['Rot_HSFO']
df['Fuj_Spread_calc'] = df['Fuj_VLSFO'] - df['Fuj_HSFO']
df['Hou_Spread_calc'] = df['Hou_VLSFO'] - df['Hou_HSFO']
df['Pan_Spread_calc'] = df['Pan_VLSFO'] - df['Pan_HSFO']

print("\nSpread verification:")
print("SG max diff:", (df['SG_Spread'] - df['SG_Spread_calc']).abs().max())
print("Rot max diff:", (df['Rot_Spread'] - df['Rot_Spread_calc']).abs().max())
print("Fuj max diff:", (df['Fuj_Spread'] - df['Fuj_Spread_calc']).abs().max())
print("Hou max diff:", (df['Hou_Spread'] - df['Hou_Spread_calc']).abs().max())
print("Pan max diff:", (df['Pan_Spread'] - df['Pan_Spread_calc']).abs().max())

# ==================================
# DESCRIPTIVE STATISTICS
# ==================================
spread_cols = ['SG_Spread', 'Rot_Spread', 'Fuj_Spread', 'Hou_Spread', 'Pan_Spread']
desc_stats = df[spread_cols].describe().T

# Add distribution metrics
desc_stats['Skewness'] = df[spread_cols].skew()
desc_stats['Kurtosis'] = df[spread_cols].kurtosis()

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS FOR FUEL SPREADS")
print("="*60)
print(desc_stats.round(4))
print("="*60)

# ==================================
# MEAN REVERSION AND NORMALITY TESTS
# ==================================
def run_statistical_tests(series, name):
    """Perform statistical tests for mean reversion and normality"""
    clean_series = series.dropna()
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(clean_series)
    
    # Jarque-Bera test
    jb_result = jarque_bera(clean_series)
    
    return {
        'Hub': name,
        'ADF Statistic': adf_result[0],
        'ADF p-value': adf_result[1],
        'Critical Value (1%)': adf_result[4]['1%'],
        'Critical Value (5%)': adf_result[4]['5%'],
        'Stationary (5% level)': adf_result[1] < 0.05,
        'Jarque-Bera Statistic': jb_result[0],
        'JB p-value': jb_result[1],
        'Normal Distribution (5% level)': jb_result[1] > 0.05
    }

# Initialize and run tests
test_results = []
for hub in ['SG', 'Rot', 'Fuj', 'Hou', 'Pan']:
    col_name = f'{hub}_Spread'
    result = run_statistical_tests(df[col_name], hub)
    test_results.append(result)

# Convert to DataFrame and print
results_df = pd.DataFrame(test_results)
results_df.set_index('Hub', inplace=True)

print("\n" + "="*60)
print("PRELIMINARY STATISTICAL TESTS FOR FUEL SPREADS")
print("="*60)
print(results_df.round(4))
print("\nKey:")
print("- Stationary: p-value < 0.05 indicates mean-reverting series")
print("- Normal Distribution: JB p-value > 0.05 suggests normality")
print("="*60)

# ==================================
# LOG RETURNS CALCULATION & VISUALIZATION
# ==================================
# Create figure for time series plot
plt.figure(figsize=(12, 8))
for hub in ['SG', 'Rot', 'Fuj', 'Hou', 'Pan']:
    plt.plot(df['Date'], df[f'{hub}_Spread'], label=hub, alpha=0.8)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Spread ($/tonne)', fontsize=12)
plt.title('VLSFO-HSFO Price Spreads (2019-2025)', fontsize=14)
plt.legend(title='Trading Hub', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Create figure for log returns histograms
plt.figure(figsize=(15, 12))

# Calculate and plot log returns for each hub
for i, hub in enumerate(['SG', 'Rot', 'Fuj', 'Hou', 'Pan'], 1):
    col_name = f'{hub}_Spread'
    returns = np.log(df[col_name]).diff().dropna()
    
    # Create subplot
    ax = plt.subplot(3, 2, i)
    
    # Plot histogram
    n, bins, patches = plt.hist(returns, bins=30, density=True, 
                                alpha=0.7, color='steelblue',
                                label='Actual Returns')
    
    # Fit normal distribution
    mu, std = norm.fit(returns)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
    
    # Add titles and labels
    plt.title(f'{hub} Spread Log Returns', fontsize=12)
    plt.xlabel('Log Return', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    
    # Add statistics to plot
    stats_text = (f'μ = {mu:.4f}\nσ = {std:.4f}\n'
                  f'Skew = {returns.skew():.2f}\nKurt = {returns.kurtosis():.2f}')
    plt.annotate(stats_text, xy=(0.05, 0.85), xycoords='axes fraction',
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add overall title
plt.suptitle('Weekly Log Returns Distribution with Normal Curve', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# ==================================
# FINAL OUTPUT
# ==================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE - SHOWING VISUALIZATIONS")
print("="*60)
print("- Close the plot windows to exit the program")
print("- Spread time series plot will show first")
print("- Returns distribution plot will show second")
print("="*60)

# Show plots 
plt.show()