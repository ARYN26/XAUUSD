#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced XAUUSD Neural Network Script
Combines Training, Testing, and Validation with advanced features
Including USD strength, interest rates, and risk sentiment indicators
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MetaTrader5 as mt5
import pytz
import pickle
import logging
import json
import warnings
import requests
import os
from io import StringIO
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_xauusd.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
LOOKBACK = 5  # Number of previous time periods to consider
SYMBOL = 'XAUUSD'  # Trading symbol
TIMEFRAME = mt5.TIMEFRAME_H1  # 1-hour timeframe

# Secondary symbols for correlation features
RELATED_SYMBOLS = ['EURUSD', 'USDJPY', 'USDCHF', 'GBPUSD']  # Major USD pairs
COMMODITIES = ['XAGUSD', 'WTICASH', 'UKOIL']  # Silver, Oil
INDICES = ['US30Cash', 'US500Cash', 'USTECHCash']  # Dow, S&P, Nasdaq proxies

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Define paths
MODEL_DIR = 'enhanced_models'
RESULTS_DIR = 'enhanced_results'
DATA_DIR = 'external_data'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Advanced training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.4
COMPLEXITY = 'high'

# Risk management parameters
KELLY_FRACTION = 0.5  # Conservative adjustment to Kelly criterion

# Enable advanced analysis features
USE_MARKET_REGIMES = True
ENSEMBLE_SIZE = 3  # Number of models in ensemble
USE_EXTERNAL_DATA = True  # Enable external data features

# Optimized feature list based on results (expanded)
OPTIMIZED_FEATURES = [
    # Base OHLCV data
    'close', 'open', 'high', 'low', 'volume',

    # Important moving averages and technical indicators
    'sma_10', 'sma_20', 'sma_50', 'ema_20',
    'close_diff', 'close_diff_pct',
    'close_diff_lag_2', 'close_diff_lag_10',
    'force_index_1', 'price_up_volume_down',
    'ichimoku_senkou_a', 'cci_20',
    'macd', 'rsi_14', 'bb_position',
    'adx', 'adx_trend_direction',
    'volatility_20', 'regime',

    # New technical features
    'atr_14', 'atr_percentage',
    'ema_200', 'price_ema_200_ratio', 'price_sma_200_ratio',
    'stoch_k', 'stoch_d', 'stoch_crossover',

    # USD strength and correlation features
    'dollar_index', 'gold_dollar_correlation',
    'gold_sp500_correlation', 'gold_treasury_correlation',

    # Interest rate related
    'real_interest_rate', 'treasury_yield_10y',
    'treasury_yield_2y', 'yield_curve',

    # Risk sentiment
    'risk_sentiment', 'vix_index', 'vix_change',

    # Market internals
    'gold_miners_ratio', 'gold_silver_ratio',
    'commodities_index',

    # Economic indicators
    'inflation_expectation', 'economic_surprise'
]


# Helper metrics functions
def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (same sign)
    """
    return tf.reduce_mean(tf.cast(tf.sign(y_true) == tf.sign(y_pred), 'float32'))


def directional_accuracy_numpy(y_true, y_pred):
    """Calculate directional accuracy for numpy arrays"""
    return np.mean((np.sign(y_true) == np.sign(y_pred)).astype(int))


# MT5 Connection Functions
def connect_to_mt5(login, password, server="MetaQuotes-Demo"):
    """
    Connect to the MetaTrader 5 terminal
    """
    if not mt5.initialize():
        logger.error("initialize() failed")
        mt5.shutdown()
        return False

    # Connect to account
    authorized = mt5.login(login, password, server)
    if not authorized:
        logger.error(f"Failed to connect to account {login}, error code: {mt5.last_error()}")
        mt5.shutdown()
        return False

    logger.info(f"Connected to account {login}")
    return True


def get_historical_data(symbol, timeframe, from_date, to_date):
    """
    Get historical data from MT5
    """
    # Convert datetime to UTC for MT5
    utc_from = from_date.astimezone(pytz.UTC)
    utc_to = to_date.astimezone(pytz.UTC)

    # Get data from MT5
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get historical data for {symbol}: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Handle different volume column names
    if 'volume' not in df.columns:
        if 'tick_volume' in df.columns:
            logger.info(f"Using 'tick_volume' for 'volume' for {symbol}")
            df['volume'] = df['tick_volume']
        elif 'real_volume' in df.columns:
            logger.info(f"Using 'real_volume' for 'volume' for {symbol}")
            df['volume'] = df['real_volume']
        else:
            logger.info(f"No volume data found for {symbol}, creating placeholder")
            df['volume'] = 1.0

    # Convert to Arizona time
    df['arizona_time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(ARIZONA_TZ)

    # Add symbol column
    df['symbol'] = symbol

    logger.info(f"Fetched {len(df)} historical data points for {symbol}")
    return df


def filter_5pm_data(df):
    """
    Filter data to only include rows at 5 PM Arizona time
    """
    df['hour'] = df['arizona_time'].dt.hour
    filtered_df = df[df['hour'] == TARGET_HOUR].copy()
    logger.info(f"Filtered to {len(filtered_df)} data points at 5 PM Arizona time")
    return filtered_df


# External Data Collection Functions
def get_dollar_index_data(start_date, end_date):
    """
    Get historical Dollar Index (DXY) data as a proxy for USD strength
    Falls back to synthetic calculation if API fails
    """
    try:
        # Try to get DXY data from an external source
        dxy_file_path = os.path.join(DATA_DIR, 'dxy_data.csv')

        # For the purpose of this example, we'll synthesize DXY data from forex pairs
        # In a real implementation, you would fetch from a data provider API
        df_eurusd = get_historical_data('EURUSD', TIMEFRAME, start_date, end_date)
        df_usdjpy = get_historical_data('USDJPY', TIMEFRAME, start_date, end_date)

        if df_eurusd is None or df_usdjpy is None:
            logger.warning("Could not get forex data for DXY calculation, creating synthetic data")
            # Create synthetic data
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
            dxy_df = pd.DataFrame({
                'time': date_range,
                'dxy': np.random.normal(100, 2, len(date_range))  # Synthetic values around 100
            })
            return dxy_df

        # Create a synthetic Dollar Index using weighted forex pairs
        # Real DXY is: 57.6% EUR, 13.6% JPY, 11.9% GBP, 9.1% CAD, 4.2% SEK, 3.6% CHF
        df_eurusd = df_eurusd[['time', 'close']].rename(columns={'close': 'eurusd'})
        df_usdjpy = df_usdjpy[['time', 'close']].rename(columns={'close': 'usdjpy'})

        # Merge on time
        merged_df = pd.merge(df_eurusd, df_usdjpy, on='time', how='inner')

        # Simple synthetic DXY calculation (inverse of EUR/USD with adjustment)
        merged_df['dxy'] = (1 / merged_df['eurusd'] * 0.576 + merged_df['usdjpy'] / 100 * 0.136) * 50

        # Keep only necessary columns
        dxy_df = merged_df[['time', 'dxy']]

        # Save for future use
        dxy_df.to_csv(dxy_file_path, index=False)

        return dxy_df

    except Exception as e:
        logger.error(f"Error getting Dollar Index data: {e}")
        # Return a DataFrame with NaN values that match the date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'dxy': np.nan
        })


def get_treasury_yield_data(start_date, end_date):
    """
    Get historical Treasury yield data (2Y and 10Y)
    For demonstration, using synthetic data
    In production, fetch from a financial data API
    """
    try:
        # In a real implementation, fetch from U.S. Treasury API or financial data provider
        # For demonstration, we'll create synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create synthetic 10-year and 2-year yield data
        np.random.seed(42)  # For reproducibility
        base_10y = 3.5  # Base 10-year yield around 3.5%
        base_2y = 3.0  # Base 2-year yield around 3.0%

        # Create trends and fluctuations
        trend = np.cumsum(np.random.normal(0, 0.02, len(date_range))) * 0.1
        daily_noise_10y = np.random.normal(0, 0.05, len(date_range))
        daily_noise_2y = np.random.normal(0, 0.04, len(date_range))

        # Generate yields
        yield_10y = base_10y + trend + daily_noise_10y
        yield_2y = base_2y + trend * 1.2 + daily_noise_2y  # 2Y typically more volatile

        # Create DataFrame
        treasury_df = pd.DataFrame({
            'date': date_range,
            'yield_10y': yield_10y,
            'yield_2y': yield_2y,
            'yield_curve': yield_10y - yield_2y  # 10Y-2Y yield curve spread
        })

        # Resample to hourly to match our OHLC data
        # Forward fill yields since they don't update every hour
        treasury_df.set_index('date', inplace=True)
        treasury_hourly = treasury_df.resample('H').ffill().reset_index()
        treasury_hourly.rename(columns={'date': 'time'}, inplace=True)

        return treasury_hourly

    except Exception as e:
        logger.error(f"Error getting Treasury yield data: {e}")
        # Return a DataFrame with NaN values
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'yield_10y': np.nan,
            'yield_2y': np.nan,
            'yield_curve': np.nan
        })


def get_real_interest_rate(treasury_df, inflation_expectation=2.5):
    """
    Calculate real interest rate (10Y yield minus inflation expectation)
    In production, fetch actual inflation expectations from a data provider
    """
    # Create a copy to avoid modifying the original
    df = treasury_df.copy()

    # Real interest rate = Nominal interest rate - Expected inflation
    df['real_interest_rate'] = df['yield_10y'] - inflation_expectation

    return df


def get_vix_data(start_date, end_date):
    """
    Get historical VIX (Volatility Index) data
    For demonstration, using synthetic data
    In production, fetch from a financial data API
    """
    try:
        # In a real implementation, fetch from financial data provider
        # For demonstration, we'll create synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create synthetic VIX data
        np.random.seed(43)  # Different seed than treasury
        base_vix = 18  # Base VIX around 18

        # Create trends and fluctuations with occasional spikes
        trend = np.cumsum(np.random.normal(0, 0.02, len(date_range))) * 0.5
        daily_noise = np.random.normal(0, 1, len(date_range))

        # Add occasional volatility spikes
        spikes = np.zeros(len(date_range))
        spike_points = np.random.choice(len(date_range), size=int(len(date_range) * 0.05), replace=False)
        spikes[spike_points] = np.random.gamma(3, 4, size=len(spike_points))

        # Generate VIX values with minimum of 9
        vix_values = np.maximum(base_vix + trend + daily_noise + spikes, 9)

        # Create DataFrame
        vix_df = pd.DataFrame({
            'date': date_range,
            'vix': vix_values
        })

        # Calculate VIX change
        vix_df['vix_change'] = vix_df['vix'].pct_change() * 100

        # Resample to hourly to match our OHLC data
        vix_df.set_index('date', inplace=True)
        vix_hourly = vix_df.resample('H').ffill().reset_index()
        vix_hourly.rename(columns={'date': 'time'}, inplace=True)

        return vix_hourly

    except Exception as e:
        logger.error(f"Error getting VIX data: {e}")
        # Return a DataFrame with NaN values
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'vix': np.nan,
            'vix_change': np.nan
        })


def get_gold_miners_ratio(gold_df, start_date, end_date):
    """
    Calculate Gold to Gold Miners ratio (using synthetic data for miners)
    In production, fetch Gold Miners ETF (GDX) data from a provider
    """
    try:
        # For demonstration, create synthetic Gold Miners ETF data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Get gold prices resampled to daily for proper ratio calculation
        gold_daily = gold_df.set_index('time').resample('D').last()['close'].reset_index()

        # Create synthetic Gold Miners ETF based on gold with different sensitivity
        np.random.seed(44)
        gdx_base = 30  # Base price for GDX
        beta = 1.5  # Miners typically have higher beta to gold

        # Get gold returns
        gold_returns = gold_daily['close'].pct_change().fillna(0)

        # Generate GDX values with beta to gold plus idiosyncratic noise
        gdx_returns = gold_returns * beta + np.random.normal(0, 0.01, len(gold_daily))
        gdx_values = gdx_base * (1 + gdx_returns).cumprod()

        # Create miners DataFrame
        miners_df = pd.DataFrame({
            'date': gold_daily['time'],
            'gdx': gdx_values
        })

        # Calculate Gold to Miners ratio
        miners_df['gold'] = gold_daily['close']
        miners_df['gold_miners_ratio'] = miners_df['gold'] / miners_df['gdx']

        # Resample to hourly
        miners_df.set_index('date', inplace=True)
        miners_hourly = miners_df.resample('H').ffill().reset_index()
        miners_hourly.rename(columns={'date': 'time'}, inplace=True)

        return miners_hourly[['time', 'gold_miners_ratio']]

    except Exception as e:
        logger.error(f"Error calculating Gold Miners ratio: {e}")
        # Return a DataFrame with NaN values
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'gold_miners_ratio': np.nan
        })


def get_gold_silver_ratio(start_date, end_date):
    """
    Calculate Gold to Silver ratio
    """
    try:
        # Get Gold and Silver data
        gold_df = get_historical_data('XAUUSD', TIMEFRAME, start_date, end_date)
        silver_df = get_historical_data('XAGUSD', TIMEFRAME, start_date, end_date)

        if gold_df is None or silver_df is None:
            raise ValueError("Could not fetch Gold or Silver data")

        # Keep only necessary columns
        gold_df = gold_df[['time', 'close']].rename(columns={'close': 'gold'})
        silver_df = silver_df[['time', 'close']].rename(columns={'close': 'silver'})

        # Merge data
        ratio_df = pd.merge(gold_df, silver_df, on='time', how='inner')

        # Calculate ratio
        ratio_df['gold_silver_ratio'] = ratio_df['gold'] / ratio_df['silver']

        return ratio_df[['time', 'gold_silver_ratio']]

    except Exception as e:
        logger.error(f"Error calculating Gold Silver ratio: {e}")
        # Return a DataFrame with NaN values
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'gold_silver_ratio': np.nan
        })


def get_risk_sentiment_index(vix_df, yield_curve_df):
    """
    Create a synthetic risk sentiment index based on VIX and yield curve
    Positive values indicate risk-on, negative values indicate risk-off
    """
    try:
        # Merge VIX and yield curve data
        sentiment_df = pd.merge(vix_df[['time', 'vix']],
                                yield_curve_df[['time', 'yield_curve']],
                                on='time', how='outer')

        # Forward fill any missing values
        sentiment_df = sentiment_df.ffill().bfill()

        # Normalize VIX (inverted, since high VIX = risk off)
        vix_mean = sentiment_df['vix'].mean()
        vix_std = sentiment_df['vix'].std()
        normalized_vix = -1 * (sentiment_df['vix'] - vix_mean) / vix_std

        # Normalize yield curve (positive curve = risk on)
        curve_mean = sentiment_df['yield_curve'].mean()
        curve_std = sentiment_df['yield_curve'].std()
        normalized_curve = (sentiment_df['yield_curve'] - curve_mean) / curve_std

        # Combined sentiment index (equal weights)
        sentiment_df['risk_sentiment'] = (normalized_vix + normalized_curve) / 2

        return sentiment_df[['time', 'risk_sentiment']]

    except Exception as e:
        logger.error(f"Error calculating risk sentiment index: {e}")
        return pd.DataFrame(columns=['time', 'risk_sentiment'])


def get_economic_surprise_index(start_date, end_date):
    """
    Create a synthetic economic surprise index
    In production, fetch from a data provider like Citigroup Economic Surprise Index
    """
    try:
        # For demonstration, create synthetic economic surprise data
        # In reality, this would come from a paid data provider
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate surprise index with autocorrelation
        np.random.seed(45)
        n = len(date_range)

        # Create AR(1) process for more realistic economic data
        phi = 0.98  # autocorrelation parameter
        w = np.random.normal(0, 0.2, n)
        surprise = np.zeros(n)

        surprise[0] = w[0]
        for t in range(1, n):
            surprise[t] = phi * surprise[t - 1] + w[t]

        # Scale to reasonable values
        surprise = surprise * 20

        # Create DataFrame
        surprise_df = pd.DataFrame({
            'date': date_range,
            'economic_surprise': surprise
        })

        # Resample to hourly
        surprise_df.set_index('date', inplace=True)
        surprise_hourly = surprise_df.resample('H').ffill().reset_index()
        surprise_hourly.rename(columns={'date': 'time'}, inplace=True)

        return surprise_hourly

    except Exception as e:
        logger.error(f"Error creating economic surprise index: {e}")
        # Return a DataFrame with NaN values
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'time': date_range,
            'economic_surprise': np.nan
        })


def calculate_correlation_features(gold_df, dfs_dict, window=20):
    """
    Calculate rolling correlations between gold and other assets
    """
    # Start with gold price
    corr_df = gold_df[['time', 'close']].copy()
    corr_df.rename(columns={'close': 'gold'}, inplace=True)

    # Ensure timezone consistency by converting to naive datetime
    if pd.api.types.is_datetime64tz_dtype(corr_df['time']):
        corr_df['time'] = corr_df['time'].dt.tz_localize(None)

    # Add other assets if available
    if 'dollar_index' in dfs_dict and not dfs_dict['dollar_index'].empty:
        dxy_df = dfs_dict['dollar_index'].copy()
        # Ensure compatible time format
        if pd.api.types.is_datetime64tz_dtype(dxy_df['time']):
            dxy_df['time'] = dxy_df['time'].dt.tz_localize(None)
        corr_df = pd.merge(corr_df, dxy_df[['time', 'dxy']], on='time', how='left')
        # Calculate rolling correlation
        corr_df['gold_dollar_correlation'] = corr_df['gold'].rolling(window=window).corr(corr_df['dxy'])

    # Add S&P 500 correlation if available (using US500Cash as proxy)
    sp500_data = None
    try:
        sp500_data = get_historical_data('US500Cash', TIMEFRAME,
                                         corr_df['time'].min(), corr_df['time'].max())
    except:
        logger.warning("Could not fetch S&P 500 data, skipping correlation")

    if sp500_data is not None:
        sp500_df = sp500_data[['time', 'close']].rename(columns={'close': 'sp500'}).copy()
        # Ensure compatible time format
        if pd.api.types.is_datetime64tz_dtype(sp500_df['time']):
            sp500_df['time'] = sp500_df['time'].dt.tz_localize(None)
        corr_df = pd.merge(corr_df, sp500_df, on='time', how='left')
        corr_df['gold_sp500_correlation'] = corr_df['gold'].rolling(window=window).corr(corr_df['sp500'])

    # Add Treasury correlation if available
    if 'treasury_yields' in dfs_dict and not dfs_dict['treasury_yields'].empty:
        treasury_df = dfs_dict['treasury_yields'].copy()
        # Ensure compatible time format
        if pd.api.types.is_datetime64tz_dtype(treasury_df['time']):
            treasury_df['time'] = treasury_df['time'].dt.tz_localize(None)
        corr_df = pd.merge(corr_df, treasury_df[['time', 'yield_10y']], on='time', how='left')
        corr_df['gold_treasury_correlation'] = corr_df['gold'].rolling(window=window).corr(corr_df['yield_10y'])

    # Fill NaN values in correlation columns
    correlation_columns = [col for col in corr_df.columns if 'correlation' in col]
    for col in correlation_columns:
        corr_df[col] = corr_df[col].fillna(0)

    return corr_df[['time'] + correlation_columns]


def add_commodities_index(gold_df, start_date, end_date):
    """
    Create a synthetic commodities index based on available commodity data
    """
    try:
        # Try to get other commodities
        oil_df = None
        try:
            oil_df = get_historical_data('WTICASH', TIMEFRAME, start_date, end_date)
            if oil_df is None:
                try:
                    oil_df = get_historical_data('UKOIL', TIMEFRAME, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Could not fetch UKOIL data: {e}")
        except Exception as e:
            logger.warning(f"Could not fetch WTICASH data: {e}")

        if oil_df is None:
            logger.info("Creating synthetic commodities index based only on gold (oil data unavailable)")
            # Create synthetic commodities index based solely on gold
            commodities_df = gold_df[['time', 'close']].copy()

            # Ensure time column is timezone-naive for consistency
            if pd.api.types.is_datetime64tz_dtype(commodities_df['time']):
                commodities_df['time'] = commodities_df['time'].dt.tz_localize(None)

            # Add some noise to create a synthetic index
            np.random.seed(46)
            noise = np.random.normal(0, 0.01, len(commodities_df))

            # Normalize gold price
            gold_min = commodities_df['close'].min()
            gold_max = commodities_df['close'].max()
            commodities_df['commodities_index'] = (commodities_df['close'] - gold_min) / (gold_max - gold_min) * 100

            # Add noise to differentiate from pure gold
            commodities_df['commodities_index'] = commodities_df['commodities_index'] * (1 + noise)

            return commodities_df[['time', 'commodities_index']]
        else:
            # Create index using gold and oil
            gold_norm = gold_df[['time', 'close']].copy()
            oil_norm = oil_df[['time', 'close']].copy()

            # Ensure time columns are timezone-naive for consistency
            if pd.api.types.is_datetime64tz_dtype(gold_norm['time']):
                gold_norm['time'] = gold_norm['time'].dt.tz_localize(None)
            if pd.api.types.is_datetime64tz_dtype(oil_norm['time']):
                oil_norm['time'] = oil_norm['time'].dt.tz_localize(None)

            # Normalize prices to 0-100 range
            gold_norm['gold_norm'] = (gold_norm['close'] - gold_norm['close'].min()) / (
                    gold_norm['close'].max() - gold_norm['close'].min()) * 100
            oil_norm['oil_norm'] = (oil_norm['close'] - oil_norm['close'].min()) / (
                    oil_norm['close'].max() - oil_norm['close'].min()) * 100

            # Merge data
            commodities_df = pd.merge(gold_norm[['time', 'gold_norm']],
                                      oil_norm[['time', 'oil_norm']],
                                      on='time', how='outer')

            # Fill missing values
            commodities_df = commodities_df.ffill().bfill()

            # Create weighted index (60% gold, 40% oil)
            commodities_df['commodities_index'] = commodities_df['gold_norm'] * 0.6 + commodities_df['oil_norm'] * 0.4

            return commodities_df[['time', 'commodities_index']]

    except Exception as e:
        logger.error(f"Error creating commodities index: {e}")
        # Create a basic fallback index using just gold
        try:
            commodities_df = gold_df[['time', 'close']].copy()

            # Ensure time column is timezone-naive
            if pd.api.types.is_datetime64tz_dtype(commodities_df['time']):
                commodities_df['time'] = commodities_df['time'].dt.tz_localize(None)

            # Simple normalization
            commodities_df['commodities_index'] = (commodities_df['close'] / commodities_df['close'].iloc[0]) * 100
            return commodities_df[['time', 'commodities_index']]
        except:
            # Last resort - empty dataframe
            return pd.DataFrame(columns=['time', 'commodities_index'])


# Feature Engineering Functions
def add_datetime_features(df):
    """
    Add cyclical datetime features for time-based patterns
    """
    # Extract datetime components
    df['day_of_week'] = df['arizona_time'].dt.dayofweek
    df['day_of_month'] = df['arizona_time'].dt.day
    df['day_of_year'] = df['arizona_time'].dt.dayofyear
    df['month'] = df['arizona_time'].dt.month
    df['quarter'] = df['arizona_time'].dt.quarter
    df['year'] = df['arizona_time'].dt.year
    df['week_of_year'] = df['arizona_time'].dt.isocalendar().week

    # Create cyclical features for time-based variables
    # Sine and cosine transformations for days of week (0-6)
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))

    # Sine and cosine transformations for months (1-12)
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))

    # Is weekend feature (binary)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Is month end/start features
    df['is_month_start'] = df['day_of_month'].apply(lambda x: 1 if x <= 3 else 0)
    df['is_month_end'] = df['day_of_month'].apply(lambda x: 1 if x >= 28 else 0)

    # US market session binary feature (9:30 AM - 4 PM ET)
    # In Arizona time during standard time, that's 7:30 AM - 2 PM
    df['us_market_session'] = ((df['arizona_time'].dt.hour >= 7) &
                               (df['arizona_time'].dt.hour < 14) &
                               (df['day_of_week'] < 5)).astype(int)

    # London session binary feature (8 AM - 4 PM GMT)
    # In Arizona time, that's 1 AM - 9 AM
    df['london_session'] = ((df['arizona_time'].dt.hour >= 1) &
                            (df['arizona_time'].dt.hour < 9) &
                            (df['day_of_week'] < 5)).astype(int)

    # Asian session binary feature (Tokyo/Sydney, approx 7 PM - 2 AM AZ time)
    df['asian_session'] = ((df['arizona_time'].dt.hour >= 19) |
                           (df['arizona_time'].dt.hour < 2) &
                           (df['day_of_week'] < 5)).astype(int)

    # Session overlap binary features (high volatility periods)
    df['london_ny_overlap'] = ((df['arizona_time'].dt.hour >= 7) &
                               (df['arizona_time'].dt.hour < 9) &
                               (df['day_of_week'] < 5)).astype(int)

    return df


def detect_market_regime(df, window=20):
    """
    Detect market regimes (trending, mean-reverting, volatile)
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std()

    # Calculate autocorrelation - negative values suggest mean reversion
    df['autocorrelation'] = df['returns'].rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x.dropna()) > 1 else np.nan, raw=False
    )

    # Calculate Hurst exponent (trend strength indicator)
    def hurst_exponent(returns, lags=range(2, 20)):
        tau = []
        std = []
        if len(returns.dropna()) < max(lags) + 1:
            return np.nan

        for lag in lags:
            # Construct a new series with lagged returns
            series_lagged = pd.Series(returns).diff(lag).dropna()
            if len(series_lagged) > 1:  # Ensure enough data
                tau.append(lag)
                std.append(np.std(series_lagged))

        if len(tau) < 2:  # Need at least 2 points for regression
            return np.nan

        # Calculate Hurst exponent from log-log regression slope
        m = np.polyfit(np.log(tau), np.log(std), 1)
        hurst = m[0] / 2.0
        return hurst

    # Apply Hurst exponent calculation on rolling window
    df['hurst'] = df['returns'].rolling(window=window * 2).apply(
        lambda x: hurst_exponent(x) if len(x.dropna()) > window else np.nan, raw=False
    )

    # Classify regimes:
    # Hurst > 0.6: Trending
    # Hurst < 0.4: Mean-reverting
    # Volatility > historical_avg*1.5: Volatile
    vol_threshold = df['volatility'].rolling(window=100).mean() * 1.5

    # Create regime flags
    df['regime_trending'] = ((df['hurst'] > 0.6) & (df['volatility'] <= vol_threshold)).astype(int)
    df['regime_mean_reverting'] = ((df['hurst'] < 0.4) & (df['volatility'] <= vol_threshold)).astype(int)
    df['regime_volatile'] = (df['volatility'] > vol_threshold).astype(int)

    # Create a composite regime indicator
    df['regime'] = 0  # Default/normal
    df.loc[df['regime_trending'] == 1, 'regime'] = 1  # Trending
    df.loc[df['regime_mean_reverting'] == 1, 'regime'] = 2  # Mean-reverting
    df.loc[df['regime_volatile'] == 1, 'regime'] = 3  # Volatile

    # Calculate regime stability (how long the current regime has persisted)
    df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    df['regime_stability'] = df['regime_change'].groupby(
        (df['regime_change'] != 0).cumsum()).cumcount()

    # Measure trend strength
    df['trend_strength'] = df['close'].diff(20).abs() / (df['high'].rolling(20).max() - df['low'].rolling(20).min())

    # Measure mean reversion potential
    df['mean_reversion_potential'] = (df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std())

    return df


def add_technical_indicators(df):
    """
    Add optimized technical analysis indicators
    """
    # Ensure we have OHLCV data
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column {col} not found")
            return df

    try:
        # Price differences and returns
        df['close_diff'] = df['close'].diff()
        df['close_diff_pct'] = df['close'].pct_change() * 100
        df['open_close_diff'] = df['close'] - df['open']
        df['open_close_diff_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['high_low_diff'] = df['high'] - df['low']
        df['high_low_diff_pct'] = (df['high'] - df['low']) / df['low'] * 100

        # Simple Moving Averages (key features from results)
        for window in [10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [20, 50, 200]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        # Price relative to moving averages
        for window in [20, 50, 200]:
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']

        # Force Index
        df['force_index_1'] = df['close'].diff(1) * df['volume']
        df['force_index_13'] = df['force_index_1'].ewm(span=13, adjust=False).mean()

        # Price-volume divergence
        df['volume_change'] = df['volume'].pct_change() * 100
        df['price_up_volume_down'] = ((df['close_diff'] > 0) & (df['volume_change'] < 0)).astype(int)
        df['price_down_volume_up'] = ((df['close_diff'] < 0) & (df['volume_change'] > 0)).astype(int)

        # Average True Range (ATR)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        df['atr_percentage'] = (df['atr_14'] / df['close']) * 100

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = abs(typical_price - ma_tp).rolling(window=20).mean()
        df['cci_20'] = (typical_price - ma_tp) / (0.015 * mean_deviation)

        # Ichimoku
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_kijun_sen'] = (high_26 + low_26) / 2

        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(26)

        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)

        df['ichimoku_chikou_span'] = df['close'].shift(-26)

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Price pattern detection for divergence
        df['price_higher_high'] = ((df['high'] > df['high'].shift(1)) &
                                   (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['price_lower_low'] = ((df['low'] < df['low'].shift(1)) &
                                 (df['low'].shift(1) < df['low'].shift(2))).astype(int)

        df['rsi_lower_high'] = ((df['rsi_14'] < df['rsi_14'].shift(1)) &
                                (df['rsi_14'].shift(1) > df['rsi_14'].shift(2))).astype(int)
        df['rsi_higher_low'] = ((df['rsi_14'] > df['rsi_14'].shift(1)) &
                                (df['rsi_14'].shift(1) < df['rsi_14'].shift(2))).astype(int)

        # RSI divergence detection
        df['rsi_higher_high'] = ((df['rsi_14'] > df['rsi_14'].shift(1)) &
                                 (df['rsi_14'].shift(1) > df['rsi_14'].shift(2))).astype(int)
        df['rsi_lower_low'] = ((df['rsi_14'] < df['rsi_14'].shift(1)) &
                               (df['rsi_14'].shift(1) < df['rsi_14'].shift(2))).astype(int)

        # Now we can calculate divergences safely
        df['bearish_divergence'] = ((df['price_higher_high'] == 1) &
                                    (df['rsi_lower_high'] == 1)).astype(int)
        df['bullish_divergence'] = ((df['price_lower_low'] == 1) &
                                    (df['rsi_higher_low'] == 1)).astype(int)

        # Bollinger Bands
        middle_band = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        df['bb_upper'] = middle_band + (std_dev * 2)
        df['bb_middle'] = middle_band
        df['bb_lower'] = middle_band - (std_dev * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Bollinger Band squeeze (potential for breakout)
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=50).quantile(0.2)

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        df['stoch_crossover'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

        # ADX (Average Directional Index)
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()

        plus_dm = np.where((high_diff > 0) & (high_diff > low_diff.abs()), high_diff, 0)
        minus_dm = np.where((low_diff < 0) & (low_diff.abs() > high_diff), low_diff.abs(), 0)

        tr = df['true_range']  # Already calculated for ATR

        smoothed_tr = tr.rolling(window=14).sum()
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=14).sum()
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=14).sum()

        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['adx'] = dx.rolling(window=14).mean()
        df['adx_plus_di'] = plus_di
        df['adx_minus_di'] = minus_di
        df['adx_trend_direction'] = np.where(plus_di > minus_di, 1, -1)
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)

        # Volatility
        df['volatility_20'] = df['close_diff_pct'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(window=100).mean()

        # Keltner Channels (similar to Bollinger Bands but using ATR)
        df['keltner_middle'] = df['ema_20']
        df['keltner_upper'] = df['keltner_middle'] + (df['atr_14'] * 2)
        df['keltner_lower'] = df['keltner_middle'] - (df['atr_14'] * 2)

        # Bollinger Band vs Keltner Channel (for squeeze detection)
        df['bb_kc_squeeze'] = ((df['bb_upper'] < df['keltner_upper']) &
                               (df['bb_lower'] > df['keltner_lower'])).astype(int)

        # Fix missing divergence-related columns if they weren't created correctly
        for col in ['rsi_lower_high', 'rsi_higher_low']:
            if col not in df.columns:
                if col == 'rsi_lower_high':
                    df[col] = ((df['rsi_14'] < df['rsi_14'].shift(1)) &
                               (df['rsi_14'].shift(1) > df['rsi_14'].shift(2))).astype(int)
                elif col == 'rsi_higher_low':
                    df[col] = ((df['rsi_14'] > df['rsi_14'].shift(1)) &
                               (df['rsi_14'].shift(1) < df['rsi_14'].shift(2))).astype(int)

        # Recalculate divergences if necessary
        if 'bearish_divergence' not in df.columns and 'price_higher_high' in df.columns and 'rsi_lower_high' in df.columns:
            df['bearish_divergence'] = ((df['price_higher_high'] == 1) &
                                        (df['rsi_lower_high'] == 1)).astype(int)

        if 'bullish_divergence' not in df.columns and 'price_lower_low' in df.columns and 'rsi_higher_low' in df.columns:
            df['bullish_divergence'] = ((df['price_lower_low'] == 1) &
                                        (df['rsi_higher_low'] == 1)).astype(int)

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return df


def add_lagged_features(df, lags=[1, 2, 3, 5, 10, 20]):
    """
    Add lagged features for selected columns
    """
    # Most important features to lag
    key_indicators = [
        'close_diff', 'close_diff_pct',
        'rsi_14', 'macd', 'bb_position',
        'adx', 'stoch_k', 'cci_20',
        'force_index_1', 'volatility_20'
    ]

    # Add lags for key indicators
    for col in key_indicators:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Add rate of change between key lags (2 and 10 from optimized features)
    df['close_diff_lag_2_1_diff'] = df['close_diff_lag_1'] - df['close_diff_lag_2']
    df['close_diff_lag_10_1_diff'] = df['close_diff_lag_1'] - df['close_diff_lag_10']

    # Add special lag combinations that might be useful
    if 'bb_position' in df.columns and 'rsi_14' in df.columns:
        # Combined indicator: oversold and at bottom of Bollinger
        df['oversold_at_support'] = ((df['bb_position'] < 0.2) & (df['rsi_14'] < 30)).astype(int)

        # Combined indicator: overbought and at top of Bollinger
        df['overbought_at_resistance'] = ((df['bb_position'] > 0.8) & (df['rsi_14'] > 70)).astype(int)

    return df


def add_target_variables(df):
    """
    Add target variables for prediction
    """
    # Calculate next period close price change
    df['next_close'] = df['close'].shift(-1)
    df['next_close_change_pct'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Add directional target (binary classification)
    df['next_direction'] = np.where(df['next_close'] > df['close'], 1, 0)

    # Add multi-timeframe targets for future periods
    for i in [2, 3, 5, 10]:
        df[f'close_future_{i}'] = df['close'].shift(-i)
        df[f'change_future_{i}_pct'] = ((df[f'close_future_{i}'] - df['close']) / df['close']) * 100

    # Add target for stop-loss optimization
    df['next_low'] = df['low'].shift(-1)
    df['next_drawdown_pct'] = ((df['close'] - df['next_low']) / df['close']) * 100

    # Add target for take-profit optimization
    df['next_high'] = df['high'].shift(-1)
    df['next_maxprofit_pct'] = ((df['next_high'] - df['close']) / df['close']) * 100

    return df


def merge_external_data(df, dfs_dict):
    """
    Merge external data with main DataFrame
    """
    merged_df = df.copy()

    # Ensure time column in main dataframe is timezone-naive
    if pd.api.types.is_datetime64tz_dtype(merged_df['time']):
        merged_df['time'] = merged_df['time'].dt.tz_localize(None)

    # Merge each external DataFrame if available
    for name, ext_df in dfs_dict.items():
        if ext_df is not None and not ext_df.empty:
            try:
                # Ensure time column exists
                if 'time' not in ext_df.columns:
                    logger.warning(f"Skipping {name} - no time column")
                    continue

                # Make a copy to avoid modifying original
                temp_df = ext_df.copy()

                # Ensure time column in external dataframe is timezone-naive
                if pd.api.types.is_datetime64tz_dtype(temp_df['time']):
                    temp_df['time'] = temp_df['time'].dt.tz_localize(None)

                # Get columns to merge
                merge_cols = [col for col in temp_df.columns if col != 'time']
                if not merge_cols:
                    continue

                # Merge
                merged_df = pd.merge(merged_df, temp_df[['time'] + merge_cols],
                                     on='time', how='left')

                # Log the merge
                logger.info(f"Merged {len(merge_cols)} columns from {name}")
            except Exception as e:
                logger.error(f"Error merging {name}: {e}")
                continue

    # Fill NaN values from external data
    # Different methods for different types of data
    numeric_cols = merged_df.select_dtypes(include=['number']).columns

    # Forward fill for most indicators
    for col in numeric_cols:
        if col not in df.columns and col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return merged_df


def prepare_features_and_targets(df, target_col='next_close_change_pct'):
    """
    Prepare features and target variables
    """
    # Drop non-feature columns
    feature_blacklist = [
        'time', 'arizona_time', 'next_close', 'hour',
        'open_time', 'tick_volume', 'spread', 'real_volume',
        'next_low', 'next_high'
    ]

    feature_df = df.drop(columns=feature_blacklist, errors='ignore')

    # Handle NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')

    # Get remaining NaN columns and fill with zeros
    nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Filling these columns with zeros: {nan_cols}")
        feature_df[nan_cols] = feature_df[nan_cols].fillna(0)

    # Separate target and features
    y = feature_df[target_col].values if target_col in feature_df.columns else np.zeros(len(feature_df))

    # Drop target columns from features
    target_cols = ['next_close_change_pct', 'next_direction',
                   'next_drawdown_pct', 'next_maxprofit_pct']
    target_cols += [f'change_future_{i}_pct' for i in [2, 3, 5, 10]]
    target_cols = [col for col in target_cols if col in feature_df.columns]

    X = feature_df.drop(columns=target_cols, errors='ignore')

    # Filter to keep only optimized features if they exist
    optimized_features = [f for f in OPTIMIZED_FEATURES if f in X.columns]
    logger.info(f"Using {len(optimized_features)} optimized features")

    # Save column names before converting to numpy
    feature_names = X.columns.tolist()

    # Select optimized features if available
    if optimized_features:
        X = X[optimized_features]
        feature_names = optimized_features

    # Convert to numpy arrays
    X = X.values

    return X, y, feature_names


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM/GRU models
    """
    X_seq, y_seq = [], []

    # Create sequences
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    X_seq_array = np.array(X_seq)
    y_seq_array = np.array(y_seq)

    return X_seq_array, y_seq_array


def time_series_split(df, train_size=0.7, val_size=0.15):
    """
    Split data respecting time order - no random shuffling
    """
    # Sort by time to ensure chronological order
    df = df.sort_values('time')

    # Calculate split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Training set: {len(train_df)} samples ({train_size * 100:.1f}%)")
    logger.info(f"Validation set: {len(val_df)} samples ({val_size * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({(1 - train_size - val_size) * 100:.1f}%)")

    return train_df, val_df, test_df


# Model Building Functions
def build_attention_lstm_model(input_shape, complexity='high', dropout_rate=0.4, learning_rate=0.001):
    """
    Build LSTM model with attention mechanism
    """
    # Define architecture complexity
    if complexity == 'low':
        units = [64, 32]
    elif complexity == 'medium':
        units = [128, 64, 32]
    else:  # high
        units = [256, 128, 64, 32]

    # Input
    input_layer = Input(shape=input_shape)

    # First LSTM layer
    lstm_1 = LSTM(units[0], return_sequences=True,
                  recurrent_dropout=dropout_rate,
                  recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(input_layer)
    norm_1 = BatchNormalization()(lstm_1)
    drop_1 = Dropout(dropout_rate)(norm_1)

    # Second LSTM layer
    lstm_2 = LSTM(units[1], return_sequences=True,
                  recurrent_dropout=dropout_rate,
                  recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(drop_1)
    norm_2 = BatchNormalization()(lstm_2)
    drop_2 = Dropout(dropout_rate)(norm_2)

    # Self-attention mechanism
    attention_layer = Attention()([drop_2, drop_2])

    # Third LSTM layer (if high complexity)
    if len(units) > 2:
        lstm_3 = LSTM(units[2], return_sequences=(len(units) > 3),
                      recurrent_dropout=dropout_rate,
                      recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(attention_layer)
        norm_3 = BatchNormalization()(lstm_3)
        drop_3 = Dropout(dropout_rate)(norm_3)

        # Fourth LSTM layer (if very high complexity)
        if len(units) > 3:
            lstm_4 = LSTM(units[3],
                          recurrent_dropout=dropout_rate,
                          recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(drop_3)
            norm_4 = BatchNormalization()(lstm_4)
            lstm_output = Dropout(dropout_rate)(norm_4)
        else:
            lstm_output = drop_3
    else:
        lstm_output = LSTM(units[-1],
                           recurrent_dropout=dropout_rate,
                           recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(attention_layer)
        lstm_output = BatchNormalization()(lstm_output)
        lstm_output = Dropout(dropout_rate)(lstm_output)

    # Dense layers
    dense_1 = Dense(max(16, units[-1] // 2), activation='relu',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(lstm_output)
    norm_d1 = BatchNormalization()(dense_1)
    drop_d1 = Dropout(dropout_rate / 2)(norm_d1)

    # Output layer
    output_layer = Dense(1)(drop_d1)

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', directional_accuracy, r2_keras]
    )

    return model


def build_cnn_lstm_model(input_shape, complexity='high', dropout_rate=0.4, learning_rate=0.001):
    """
    Build CNN-LSTM model (best model from results)
    """
    # Define architecture based on complexity
    if complexity == 'low':
        conv_filters = [32, 16]
        lstm_units = [64, 32]
    elif complexity == 'medium':
        conv_filters = [64, 32, 16]
        lstm_units = [128, 64]
    elif complexity == 'high':
        conv_filters = [128, 64, 32]
        lstm_units = [256, 128]
    else:
        conv_filters = [64, 32, 16]
        lstm_units = [128, 64]

    # Check the sequence length (temporal dimension)
    seq_length = input_shape[0]

    # Input layer
    input_layer = Input(shape=input_shape)

    # CNN layers
    conv_1 = Conv1D(filters=conv_filters[0], kernel_size=min(3, seq_length - 1),
                    padding='same', activation='relu')(input_layer)
    conv_1 = BatchNormalization()(conv_1)

    # Only use pooling if we have enough sequence length
    if seq_length >= 4:
        conv_1 = MaxPooling1D(pool_size=2, padding='same')(conv_1)

    conv_1 = Dropout(dropout_rate)(conv_1)

    # Second CNN layer
    if seq_length >= 3:  # Need at least 3 time steps for a kernel size of 2
        conv_2 = Conv1D(filters=conv_filters[1], kernel_size=2,
                        padding='same', activation='relu')(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Dropout(dropout_rate)(conv_2)
    else:
        conv_2 = conv_1

    # LSTM layers
    lstm_1 = LSTM(lstm_units[0], return_sequences=(len(lstm_units) > 1),
                  recurrent_dropout=dropout_rate,
                  recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(conv_2)
    lstm_1 = BatchNormalization()(lstm_1)
    lstm_1 = Dropout(dropout_rate)(lstm_1)

    if len(lstm_units) > 1:
        lstm_2 = LSTM(lstm_units[1],
                      recurrent_dropout=dropout_rate,
                      recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(lstm_1)
        lstm_2 = BatchNormalization()(lstm_2)
        lstm_output = Dropout(dropout_rate)(lstm_2)
    else:
        lstm_output = lstm_1

    # Dense layers
    dense_1 = Dense(max(16, lstm_units[-1] // 2), activation='relu',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(lstm_output)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Dropout(dropout_rate)(dense_1)

    # Output layer
    output_layer = Dense(1)(dense_1)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', directional_accuracy, r2_keras]
    )

    return model


def build_ensemble_models(input_shape, n_models=3):
    """
    Build an ensemble of models with diverse architectures
    """
    models = []

    # Base CNN-LSTM model (best from results)
    cnn_lstm = build_cnn_lstm_model(
        input_shape=input_shape,
        complexity=COMPLEXITY,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    models.append(cnn_lstm)

    # Add an Attention LSTM model for diversity
    if n_models > 1:
        attention_lstm = build_attention_lstm_model(
            input_shape=input_shape,
            complexity=COMPLEXITY,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE
        )
        models.append(attention_lstm)

    # Add a Bidirectional LSTM model for diversity
    if n_models > 2:
        bi_lstm = Sequential()

        # First Bidirectional LSTM layer
        bi_lstm.add(Bidirectional(
            LSTM(128, recurrent_dropout=DROPOUT_RATE,
                 recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5),
                 return_sequences=True),
            input_shape=input_shape
        ))
        bi_lstm.add(BatchNormalization())
        bi_lstm.add(Dropout(DROPOUT_RATE))

        # Second Bidirectional LSTM layer
        bi_lstm.add(Bidirectional(
            LSTM(64, recurrent_dropout=DROPOUT_RATE,
                 recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))
        ))
        bi_lstm.add(BatchNormalization())
        bi_lstm.add(Dropout(DROPOUT_RATE))

        # Dense layers
        bi_lstm.add(Dense(32, activation='relu'))
        bi_lstm.add(BatchNormalization())
        bi_lstm.add(Dropout(DROPOUT_RATE / 2))
        bi_lstm.add(Dense(1))

        # Compile
        bi_lstm.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

        models.append(bi_lstm)

    return models


def ensemble_predict(models, weights, X):
    """
    Make predictions with ensemble model using weighted average
    """
    predictions = []

    # Get predictions from each model
    for model in models:
        pred = model.predict(X, verbose=0)
        predictions.append(pred)

    # Weighted average
    weighted_preds = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_preds += weights[i] * pred

    return weighted_preds.flatten()


# Training and Evaluation Functions
def train_models(models, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    """
    Train all models in the ensemble
    """
    trained_models = []
    model_histories = []

    for i, model in enumerate(models):
        logger.info(f"Training model {i + 1}/{len(models)}")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f'model_{i}.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        trained_models.append(model)
        model_histories.append(history)

    return trained_models, model_histories


def evaluate_ensemble(models, X_test_seq, y_test_seq):
    """
    Evaluate the ensemble model
    """
    # Use equal weights initially
    weights = [1.0 / len(models)] * len(models)

    # Make predictions with each model
    individual_preds = []
    individual_metrics = []

    for i, model in enumerate(models):
        preds = model.predict(X_test_seq).flatten()
        individual_preds.append(preds)

        # Calculate metrics
        mse = mean_squared_error(y_test_seq, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_seq, preds)
        r2 = r2_score(y_test_seq, preds)
        dir_acc = directional_accuracy_numpy(y_test_seq, preds)

        logger.info(f"Model {i + 1} Test Metrics:")
        logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}, R²: {r2:.6f}")
        logger.info(f"Directional Accuracy: {dir_acc:.4f}")

        individual_metrics.append({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': dir_acc
        })

        # Update weights based on MSE (lower is better)
        weights[i] = 1.0 / max(mse, 1e-10)  # Avoid division by zero

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    logger.info(f"Ensemble weights: {normalized_weights}")

    # Make ensemble prediction
    ensemble_preds = ensemble_predict(models, normalized_weights, X_test_seq)

    # Calculate ensemble metrics
    mse = mean_squared_error(y_test_seq, ensemble_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, ensemble_preds)
    r2 = r2_score(y_test_seq, ensemble_preds)
    dir_acc = directional_accuracy_numpy(y_test_seq, ensemble_preds)

    logger.info(f"Ensemble Test Metrics:")
    logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}, R²: {r2:.6f}")
    logger.info(f"Directional Accuracy: {dir_acc:.4f}")

    ensemble_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': dir_acc,
        'individual_metrics': individual_metrics,
        'weights': normalized_weights
    }

    return ensemble_preds, ensemble_metrics, normalized_weights


def calculate_feature_importance(model, X_test_seq, y_test_seq, feature_list):
    """
    Calculate feature importance using permutation method
    """
    # Baseline performance
    y_pred = model.predict(X_test_seq, verbose=0)
    baseline_mse = mean_squared_error(y_test_seq, y_pred)

    # Calculate importance for each feature
    importances = []

    # Loop through each feature
    for i in range(X_test_seq.shape[2]):
        # Copy the data
        X_permuted = X_test_seq.copy()

        # Shuffle the feature across all sequences
        for j in range(X_permuted.shape[0]):
            np.random.shuffle(X_permuted[j, :, i])

        # Predict and calculate MSE
        y_pred_permuted = model.predict(X_permuted, verbose=0)
        permuted_mse = mean_squared_error(y_test_seq, y_pred_permuted)

        # Importance is the increase in error
        importance = permuted_mse - baseline_mse
        importances.append(importance)

        logger.info(f"Feature {feature_list[i]} importance: {importance:.6f}")

    # Normalize importances
    if max(importances) > 0:
        normalized_importances = [imp / max(importances) for imp in importances]
    else:
        normalized_importances = importances

    # Create DataFrame with features and importances
    importance_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': importances,
        'Normalized_Importance': normalized_importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df


def calculate_kelly_criterion(predictions, actuals, position_type='long_short'):
    """
    Calculate Kelly criterion for optimal position sizing
    position_type: 'long_short' for both directions, 'long_only' or 'short_only'
    """
    # Ensure predictions and actuals are 1D arrays
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter predictions based on position type
    if position_type == 'long_only':
        mask = predictions > 0
        if not any(mask):
            return 0.0  # No valid long trades
        pred_filtered = predictions[mask]
        actual_filtered = actuals[mask]
    elif position_type == 'short_only':
        mask = predictions < 0
        if not any(mask):
            return 0.0  # No valid short trades
        pred_filtered = -predictions[mask]  # Flip sign for shorts
        actual_filtered = -actuals[mask]  # Flip sign for shorts
    else:  # long_short
        pred_filtered = np.abs(predictions)
        actual_filtered = actuals * np.sign(predictions)  # Align with prediction directions

    if len(pred_filtered) == 0:
        return 0.0  # No valid trades of this type

    # Calculate win probability and average win/loss
    wins = actual_filtered > 0
    win_prob = np.mean(wins)

    if win_prob == 0:
        return 0.0  # No winning trades

    avg_win = np.mean(actual_filtered[wins]) if any(wins) else 0
    avg_loss = np.abs(np.mean(actual_filtered[~wins])) if any(~wins) else 0

    if avg_loss == 0:
        return 1.0  # No losing trades

    # Calculate Kelly fraction
    try:
        kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
    except:
        return 0.0  # Error in calculation

    # Limit to reasonable range
    kelly = max(0, min(1, kelly))

    return kelly


def calculate_risk_metrics(predictions, actuals, regime_data=None):
    """
    Calculate risk management metrics
    """
    # Basic Kelly criterion for different position types
    kelly_long = calculate_kelly_criterion(predictions, actuals, 'long_only')
    kelly_short = calculate_kelly_criterion(predictions, actuals, 'short_only')
    kelly_combined = calculate_kelly_criterion(predictions, actuals, 'long_short')

    # Apply KELLY_FRACTION to get conservative position sizing
    position_size_long = kelly_long * KELLY_FRACTION
    position_size_short = kelly_short * KELLY_FRACTION
    position_size_combined = kelly_combined * KELLY_FRACTION

    risk_metrics = {
        'kelly_long': kelly_long,
        'kelly_short': kelly_short,
        'kelly_combined': kelly_combined,
        'position_size_long': position_size_long,
        'position_size_short': position_size_short,
        'position_size_combined': position_size_combined
    }

    logger.info(f"Kelly Position Sizing:")
    logger.info(f"Long: {position_size_long:.2%}")
    logger.info(f"Short: {position_size_short:.2%}")
    logger.info(f"Combined: {position_size_combined:.2%}")

    # Calculate regime-specific performance if regime data available
    if regime_data is not None and 'regime' in regime_data.columns:
        regime_kelly = {}

        # Filter each regime
        for regime in range(4):  # 0: normal, 1: trending, 2: mean-reverting, 3: volatile
            regime_mask = regime_data['regime'] == regime

            if sum(regime_mask) >= 10:  # Need at least 10 samples for meaningful calculation
                # Calculate regime-specific Kelly
                regime_predictions = predictions[regime_mask]
                regime_actuals = actuals[regime_mask]

                # Directional accuracy for this regime
                regime_dir_acc = directional_accuracy_numpy(regime_actuals, regime_predictions)

                # Calculate Kelly for this regime
                regime_kelly[f'regime_{regime}_kelly'] = calculate_kelly_criterion(
                    regime_predictions, regime_actuals)
                regime_kelly[f'regime_{regime}_position_size'] = regime_kelly[f'regime_{regime}_kelly'] * KELLY_FRACTION
                regime_kelly[f'regime_{regime}_dir_acc'] = regime_dir_acc

                logger.info(f"Regime {regime} Position Size: {regime_kelly[f'regime_{regime}_position_size']:.2%}")
                logger.info(f"Regime {regime} Dir Accuracy: {regime_dir_acc:.2%}")

        risk_metrics.update(regime_kelly)

    return risk_metrics


def plot_results(models, X_test_seq, ensemble_preds, y_test, test_dates, histories, feature_importance=None,
                 regime_data=None):
    """
    Plot and save results visualization
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Create a DataFrame with results for easier plotting
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test,
        'Ensemble_Prediction': ensemble_preds
    })

    # Add individual model predictions if available
    for i, model in enumerate(models):
        preds = model.predict(X_test_seq, verbose=0).flatten()
        results_df[f'Model_{i + 1}_Prediction'] = preds

    # Add regime data if available
    if regime_data is not None and 'regime' in regime_data.columns:
        results_df['Regime'] = regime_data['regime'].values

    # 1. Plot training history
    plt.figure(figsize=(15, 10))

    # Plot loss for each model
    plt.subplot(2, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Model {i + 1} Train')
        plt.plot(history.history['val_loss'], label=f'Model {i + 1} Val', linestyle='--')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot directional accuracy for each model
    plt.subplot(2, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['directional_accuracy'], label=f'Model {i + 1} Train')
        plt.plot(history.history['val_directional_accuracy'], label=f'Model {i + 1} Val', linestyle='--')
    plt.title('Directional Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot predictions vs actual
    plt.subplot(2, 2, 3)
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual', linewidth=2)
    plt.plot(results_df['Date'], results_df['Ensemble_Prediction'], label='Ensemble', linewidth=2)
    plt.title('Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price Change (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot error
    plt.subplot(2, 2, 4)
    error = results_df['Actual'] - results_df['Ensemble_Prediction']
    plt.plot(results_df['Date'], error)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Prediction Error')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_results.png'))

    # 2. Plot scatter of predicted vs actual
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['Actual'], results_df['Ensemble_Prediction'], alpha=0.5)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Add 45-degree line
    min_val = min(results_df['Actual'].min(), results_df['Ensemble_Prediction'].min())
    max_val = max(results_df['Actual'].max(), results_df['Ensemble_Prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR, 'scatter_plot.png'))

    # 3. Plot feature importance if provided
    if feature_importance is not None:
        plt.figure(figsize=(12, 10))
        top_n = min(25, len(feature_importance))
        top_features = feature_importance.head(top_n)

        plt.barh(top_features['Feature'], top_features['Normalized_Importance'])
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Normalized Importance')
        plt.gca().invert_yaxis()  # Display highest importance at the top
        plt.grid(True, axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))

    # 4. Plot cumulative returns comparison
    plt.figure(figsize=(15, 7))

    # Calculate returns based on predictions
    results_df['Strategy_Return'] = np.sign(results_df['Ensemble_Prediction']) * results_df['Actual']
    results_df['Buy_Hold_Return'] = results_df['Actual']

    # Calculate cumulative returns
    results_df['Cumulative_Strategy'] = (1 + results_df['Strategy_Return'] / 100).cumprod() - 1
    results_df['Cumulative_Buy_Hold'] = (1 + results_df['Buy_Hold_Return'] / 100).cumprod() - 1

    plt.plot(results_df['Date'], results_df['Cumulative_Strategy'] * 100, label='Model Strategy', linewidth=2)
    plt.plot(results_df['Date'], results_df['Cumulative_Buy_Hold'] * 100, label='Buy & Hold', linewidth=2)

    plt.title('Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cumulative_returns.png'))

    # 5. Plot regime-based performance if regime data is available
    if regime_data is not None and 'regime' in regime_data.columns:
        plt.figure(figsize=(15, 12))

        # Plot regime distribution
        plt.subplot(2, 2, 1)
        regime_counts = regime_data['regime'].value_counts().sort_index()
        regime_labels = ['Normal', 'Trending', 'Mean-Rev', 'Volatile']
        plt.bar(regime_labels[:len(regime_counts)], regime_counts)
        plt.title('Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.grid(True, axis='y')

        # Plot directional accuracy by regime
        plt.subplot(2, 2, 2)
        dir_acc_by_regime = []

        for regime in range(min(4, regime_counts.index.max() + 1)):
            regime_mask = results_df['Regime'] == regime
            if sum(regime_mask) > 0:
                regime_dir_acc = directional_accuracy_numpy(
                    results_df.loc[regime_mask, 'Actual'],
                    results_df.loc[regime_mask, 'Ensemble_Prediction']
                )
                dir_acc_by_regime.append(regime_dir_acc)
            else:
                dir_acc_by_regime.append(0)

        plt.bar(regime_labels[:len(dir_acc_by_regime)], dir_acc_by_regime)
        plt.title('Directional Accuracy by Regime')
        plt.xlabel('Regime')
        plt.ylabel('Accuracy')
        plt.grid(True, axis='y')
        plt.axhline(y=0.5, color='r', linestyle='--')

        # Plot returns by regime
        plt.subplot(2, 2, 3)
        returns_by_regime = []

        for regime in range(min(4, regime_counts.index.max() + 1)):
            regime_mask = results_df['Regime'] == regime
            if sum(regime_mask) > 0:
                regime_returns = np.mean(results_df.loc[regime_mask, 'Strategy_Return'])
                returns_by_regime.append(regime_returns)
            else:
                returns_by_regime.append(0)

        plt.bar(regime_labels[:len(returns_by_regime)], returns_by_regime)
        plt.title('Average Strategy Return by Regime (%)')
        plt.xlabel('Regime')
        plt.ylabel('Return (%)')
        plt.grid(True, axis='y')
        plt.axhline(y=0, color='r', linestyle='--')

        # Plot regime over time with predictions
        plt.subplot(2, 2, 4)
        plt.plot(results_df['Date'], results_df['Actual'], label='Actual', alpha=0.6)
        plt.plot(results_df['Date'], results_df['Ensemble_Prediction'], label='Prediction', alpha=0.6)

        # Add background colors for regimes
        for regime, color in zip(range(4), ['lightgray', 'lightgreen', 'lightblue', 'salmon']):
            regime_mask = results_df['Regime'] == regime
            if not any(regime_mask):
                continue

            # Extract spans of consecutive same-regime periods
            regime_changes = results_df['Regime'].ne(results_df['Regime'].shift()).cumsum()
            for group_id, group in results_df.groupby(regime_changes):
                if group['Regime'].iloc[0] == regime:  # Only plot spans for the current regime
                    plt.axvspan(group['Date'].iloc[0], group['Date'].iloc[-1], alpha=0.2, color=color)

        plt.title('Price Changes and Regimes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price Change (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'regime_analysis.png'))

    # Save results DataFrame to CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, 'test_results.csv'), index=False)


# Data Pipeline
def load_and_preprocess_data(login, password, server="MetaQuotes-Demo"):
    """
    Load and preprocess data from MT5 with external features
    """
    # Connect to MT5
    if not connect_to_mt5(login, password, server):
        logger.error("Failed to connect to MT5")
        return None

    try:
        # Define date range for training data (3 years)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=3 * 365)

        # Get historical data
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        gold_df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)
        if gold_df is None:
            logger.error("Failed to get historical gold data")
            return None

        # Get related currency data (if available)
        currency_dfs = {}
        for symbol in RELATED_SYMBOLS:
            try:
                df = get_historical_data(symbol, TIMEFRAME, start_date, end_date)
                if df is not None:
                    currency_dfs[symbol] = df
                    logger.info(f"Fetched {len(df)} data points for {symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch {symbol} data: {e}")

        # Get commodity data (if available)
        commodity_dfs = {}
        for symbol in COMMODITIES:
            try:
                df = get_historical_data(symbol, TIMEFRAME, start_date, end_date)
                if df is not None:
                    commodity_dfs[symbol] = df
                    logger.info(f"Fetched {len(df)} data points for {symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch {symbol} data: {e}")

        # Get index data (if available)
        index_dfs = {}
        for symbol in INDICES:
            try:
                df = get_historical_data(symbol, TIMEFRAME, start_date, end_date)
                if df is not None:
                    index_dfs[symbol] = df
                    logger.info(f"Fetched {len(df)} data points for {symbol}")
            except Exception as e:
                logger.warning(f"Could not fetch {symbol} data: {e}")

        # Filter for 5 PM Arizona time or use all data if not enough
        gold_df_5pm = filter_5pm_data(gold_df)

        # Choose dataset based on availability
        if len(gold_df_5pm) >= 200:  # We need at least 200 data points
            logger.info(f"Using 5 PM Arizona time data with {len(gold_df_5pm)} data points")
            df_processed = gold_df_5pm
        else:
            logger.warning(f"Not enough 5 PM data (only {len(gold_df_5pm)}), using all hours")
            df_processed = gold_df

        # Get external data if enabled
        external_data = {}
        if USE_EXTERNAL_DATA:
            logger.info("Getting external data features...")

            # Get USD strength data
            logger.info("Fetching Dollar Index data...")
            dxy_df = get_dollar_index_data(start_date, end_date)
            external_data['dollar_index'] = dxy_df

            # Get Treasury yield and real interest rate data
            logger.info("Fetching Treasury yield data...")
            treasury_df = get_treasury_yield_data(start_date, end_date)
            treasury_df_with_real = get_real_interest_rate(treasury_df)
            external_data['treasury_yields'] = treasury_df_with_real

            # Get VIX data for risk sentiment
            logger.info("Fetching VIX data...")
            vix_df = get_vix_data(start_date, end_date)
            external_data['vix'] = vix_df

            # Get risk sentiment index
            logger.info("Calculating risk sentiment index...")
            risk_sentiment_df = get_risk_sentiment_index(vix_df, treasury_df)
            external_data['risk_sentiment'] = risk_sentiment_df

            # Get Gold Miners ratio
            logger.info("Calculating Gold Miners ratio...")
            miners_df = get_gold_miners_ratio(gold_df, start_date, end_date)
            external_data['gold_miners'] = miners_df

            # Get Gold Silver ratio
            logger.info("Calculating Gold Silver ratio...")
            gold_silver_df = get_gold_silver_ratio(start_date, end_date)
            external_data['gold_silver_ratio'] = gold_silver_df

            # Get economic surprise index
            logger.info("Creating economic surprise index...")
            econ_surprise_df = get_economic_surprise_index(start_date, end_date)
            external_data['economic_surprise'] = econ_surprise_df

            # Get commodities index
            logger.info("Creating commodities index...")
            commodities_df = add_commodities_index(gold_df, start_date, end_date)
            external_data['commodities_index'] = commodities_df

            # Calculate correlations between gold and other assets
            logger.info("Calculating correlation features...")
            correlations_df = calculate_correlation_features(gold_df, external_data)
            external_data['correlations'] = correlations_df

        # Add datetime features
        logger.info("Adding datetime features...")
        df_processed = add_datetime_features(df_processed)

        # Add technical indicators
        logger.info("Adding technical indicators...")
        df_processed = add_technical_indicators(df_processed)

        # Detect market regimes
        if USE_MARKET_REGIMES:
            logger.info("Detecting market regimes...")
            df_processed = detect_market_regime(df_processed)

        # Add lagged features
        logger.info("Adding lagged features...")
        df_processed = add_lagged_features(df_processed)

        # Add target variables
        logger.info("Adding target variables...")
        df_processed = add_target_variables(df_processed)

        # Merge external data with main dataframe
        if USE_EXTERNAL_DATA and external_data:
            logger.info("Merging external data...")
            df_processed = merge_external_data(df_processed, external_data)

        return df_processed

    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        logger.info("MT5 connection closed")


def save_models_and_results(models, ensemble_metrics, feature_importance, risk_metrics, normalized_weights):
    """
    Save models, metrics, and results to files
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save metrics to JSON
    metrics_file = os.path.join(RESULTS_DIR, 'metrics.json')
    metrics_to_save = {
        'ensemble_metrics': {
            'mse': ensemble_metrics['mse'],
            'rmse': ensemble_metrics['rmse'],
            'mae': ensemble_metrics['mae'],
            'r2': ensemble_metrics['r2'],
            'directional_accuracy': ensemble_metrics['directional_accuracy']
        },
        'model_weights': normalized_weights,
        'risk_metrics': risk_metrics
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=4)

    # Save feature importance
    importance_file = os.path.join(RESULTS_DIR, 'feature_importance.csv')
    feature_importance.to_csv(importance_file, index=False, encoding='utf-8')

    # Save model architecture summaries
    with open(os.path.join(RESULTS_DIR, 'model_architectures.txt'), 'w', encoding='utf-8') as f:
        for i, model in enumerate(models):
            f.write(f"Model {i + 1} Architecture:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n\n")


def main(login, password, server):
    """
    Main function to run the entire pipeline
    """
    logger.info("Starting Enhanced XAUUSD Neural Network Script")

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = load_and_preprocess_data(login, password, server)
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Split into train, validation, and test sets
    logger.info("Splitting data into train/val/test sets...")
    train_df, val_df, test_df = time_series_split(data)

    # Prepare features and targets for training
    logger.info("Preparing features and targets...")
    X_train, y_train, feature_names = prepare_features_and_targets(train_df)
    X_val, y_val, _ = prepare_features_and_targets(val_df)
    X_test, y_test, _ = prepare_features_and_targets(test_df)

    # Create sequences for LSTM/CNN models
    logger.info("Creating sequences for deep learning models...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, LOOKBACK)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, LOOKBACK)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, LOOKBACK)

    # Get dates for plotting
    test_dates = test_df['time'].iloc[LOOKBACK:].values

    # Build ensemble models
    logger.info(f"Building ensemble of {ENSEMBLE_SIZE} models...")
    input_shape = (LOOKBACK, X_train.shape[1])
    models = build_ensemble_models(input_shape, ENSEMBLE_SIZE)

    # Train models
    logger.info("Training models...")
    trained_models, histories = train_models(models, X_train_seq, y_train_seq, X_val_seq, y_val_seq)

    # Evaluate ensemble on test data
    logger.info("Evaluating models on test data...")
    ensemble_preds, ensemble_metrics, normalized_weights = evaluate_ensemble(trained_models, X_test_seq, y_test_seq)

    # Calculate feature importance
    logger.info("Calculating feature importance...")
    feature_importance = calculate_feature_importance(trained_models[0], X_test_seq, y_test_seq, feature_names)

    # Calculate risk metrics
    logger.info("Calculating risk metrics...")
    regime_data = test_df.iloc[LOOKBACK:][['regime']] if 'regime' in test_df.columns else None
    risk_metrics = calculate_risk_metrics(ensemble_preds, y_test_seq, regime_data)

    # Plot results
    logger.info("Plotting results...")
    plot_results(trained_models, X_test_seq, ensemble_preds, y_test_seq, test_dates, histories,
                 feature_importance, regime_data)

    # Save models and results
    logger.info("Saving models and results...")
    save_models_and_results(trained_models, ensemble_metrics, feature_importance, risk_metrics, normalized_weights)

    logger.info("Script completed successfully")
    return trained_models, ensemble_metrics, feature_importance, risk_metrics


if __name__ == "__main__":
    # MT5 Account credentials
    LOGIN = 90933473
    PASSWORD = "NhXgR*3g"
    SERVER = "MetaQuotes-Demo"

    # Run the main function
    main(LOGIN, PASSWORD, SERVER)