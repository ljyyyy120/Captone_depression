import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def adf_test_on_detrended_sami(data, cbsa_column='cbsacode', sami_column='SSAMI_detrended'):
    """
    Conducts the ADF test on the detrended SAMI time series for each CBSA, assessing stationarity.
    
    Parameters:
    - data: DataFrame containing the detrended SAMI time series.
    - cbsa_column: Column name for CBSA codes.
    - sami_column: Column name for detrended SAMI values.
    
    Returns:
    - A DataFrame with ADF test results for each CBSA.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected data to be a DataFrame")
        
    results = []
    cbsa_codes = data[cbsa_column].unique()

    for cbsa in cbsa_codes:
        # Extract data for the current CBSA
        cbsa_data = pd.to_numeric(data[data[cbsa_column] == cbsa].set_index('year')[sami_column].sort_index(), errors='coerce')
        
        # Check if the variance is not zero for the SAMI series (avoid constant series)
        if np.var(cbsa_data.dropna()) > 1e-6:  # Small threshold for near-constant
            try:
                # Suppress warnings temporarily within this block
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adf_result = adfuller(cbsa_data.dropna())
                results.append({
                    'CBSA': cbsa,
                    'ADF Statistic': adf_result[0],
                    'p-value': adf_result[1],
                    'Critical Values': adf_result[4],
                    'Stationary': adf_result[1] < 0.05  # Stationary if p-value < 0.05
                })
            except ValueError:
                results.append({
                    'CBSA': cbsa,
                    'ADF Statistic': None,
                    'p-value': None,
                    'Critical Values': 'Insufficient data for ADF test',
                    'Stationary': False
                })
        else:
            results.append({
                'CBSA': cbsa,
                'ADF Statistic': None,
                'p-value': None,
                'Critical Values': 'Constant or near-constant data',
                'Stationary': False
            })

    # Convert the results to a DataFrame
    return pd.DataFrame(results)

def stationarity_test(data, factor_column, cbsa_column='cbsacode', sami_column='SSAMI_detrended'):
    """
    Calculates ADF test results for detrended SAMI time series for each CBSA and each factor.
    
    Parameters:
    - data: DataFrame containing the detrended SAMI time series.
    - factor_column: Column indicating the factors.
    - cbsa_column: Column name for CBSA codes.
    - sami_column: Column name for detrended SAMI values.
    
    Returns:
    - A DataFrame with stationarity test results for each factor and CBSA.
    """
    results = []

    # Loop through each factor
    factors = data[factor_column].unique()
    for factor in factors:
        print(f"Processing factor: {factor}")
        
        # Filter data for the current factor
        factor_data = data[data[factor_column] == factor]
        
        # Perform the ADF test on the detrended SAMI data for each CBSA
        adf_results = adf_test_on_detrended_sami(factor_data, cbsa_column, sami_column)
        
        # Add the factor name to the results
        adf_results['Factor'] = factor
        results.append(adf_results)

    # Combine results for all factors into a single DataFrame
    final_results_df = pd.concat(results, ignore_index=True)
    return final_results_df

def arima_detrend(data, cbsa_column='cbsacode', sami_column='SSAMI', order=(1, 1, 4)):
    """
    Detrends SAMI time series using ARIMA model residuals for each CBSA.
    
    Parameters:
    - data: DataFrame containing SAMI data.
    - cbsa_column: Column name for CBSA codes.
    - sami_column: Column name for SAMI values.
    - order: Tuple (p, d, q) specifying the ARIMA model order.
    
    Returns:
    - A DataFrame with a new column 'SAMI_detrended_arima'.
    """
    # Ensure the SAMI column is numeric
    data[sami_column] = pd.to_numeric(data[sami_column], errors='coerce')
    
    # Initialize a new column for detrended data
    data['SAMI_detrended_arima'] = None
    
    # Group by CBSA code
    grouped = data.groupby(cbsa_column)
    
    for cbsa, group in grouped:
        sami_values = group[sami_column].values
        years = group['year'].values
        
        # Remove NaN values from the SAMI series
        if np.isnan(sami_values).any():
            print(f"Skipping CBSA {cbsa} due to NaN values.")
            continue
        
        # Check if there are enough data points for ARIMA
        if len(sami_values) > sum(order):  # Ensure sufficient data points
            try:
                # Fit ARIMA model
                model = ARIMA(sami_values, order=order)
                fitted_model = model.fit()
                
                # Use residuals as detrended series
                residuals = fitted_model.resid
                data.loc[group.index, 'SAMI_detrended_arima'] = residuals
            except Exception as e:
                print(f"ARIMA fitting failed for CBSA {cbsa}: {e}")
        else:
            print(f"Skipping CBSA {cbsa} due to insufficient data points.")
    
    return data