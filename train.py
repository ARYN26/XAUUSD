#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Training Script with Improved Architecture
Focus: 5 PM Arizona Time Data with Advanced Feature Engineering, Ensemble Techniques and Day-Specific Models
Implements findings from academic research papers on financial forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate, Add, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_regression
import MetaTrader5 as mt5
import pytz
import pickle
import logging
from scipy import stats
import warnings
import tensorflow.keras.backend as K
import json
from scipy.stats import skew, kurtosis
import gc
from collections import defaultdict
from tqdm import tqdm

try:
    import optuna
    from optuna import create_study
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
LOOKBACK = 5  # Number of previous time periods to consider
SYMBOL = 'XAUUSD'  # Trading symbol - Changed to Gold/USD as per repo name
TIMEFRAME = mt5.TIMEFRAME_H1  # 1-hour timeframe

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Define paths
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
RESULTS_DIR = 'Results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Day-specific model paths
DAY_MODEL_DIR = os.path.join(MODEL_DIR, 'day_models')
os.makedirs(DAY_MODEL_DIR, exist_ok=True)

# Model type options (adding simpler models first)
MODEL_TYPES = ['random_forest', 'gbm', 'simple_nn', 'lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention']

# Additional Parameters
ENSEMBLE_SIZE = 5  # Increased from 3 to 5 for better ensemble diversity
USE_WAVELET = True  # Whether to use wavelet transformation for feature extraction
MARKET_REGIMES = True  # Whether to detect market regimes
USE_DAY_SPECIFIC_MODELS = True  # Train separate models for each day of week
USE_FEATURE_SELECTION = True  # Whether to use feature selection
FEATURE_SELECTION_METHOD = 'rfe'  # Options: 'random_forest', 'correlation', 'rfe', 'mutual_info'
MIN_FEATURES = 30  # Minimum number of features to keep
USE_HYPEROPT = OPTUNA_AVAILABLE  # Use Optuna for hyperparameter optimization if available
SIMPLER_MODELS_FIRST = True  # Try simpler models before complex ones
ADD_MACROECONOMIC = True  # Add macroeconomic features that impact gold
HANDLE_OUTLIERS = True  # Handle outliers in the data
WEIGHTED_LOSS = True  # Use weighted loss function to focus on larger moves
USE_SMOTE = SMOTE_AVAILABLE  # Use SMOTE for synthetic data generation if available
CONFIDENCE_THRESHOLD = 0.6  # Threshold for trade confidence


# Import necessary external data (simulated for this implementation)
def load_economic_indicators():
    """
    Load or create economic indicators relevant to gold prices
    In real implementation, this would load from actual sources or APIs
    """
    # Generate mock data for demonstration
    dates = pd.date_range(start='2020-01-01', end='2025-01-01', freq='D')
    n_dates = len(dates)

    # Create economic indicators dataframe
    indicators = pd.DataFrame({
        'date': dates,
        'usd_index': np.random.normal(90, 5, n_dates).cumsum() / 100,  # USD Index
        'inflation_rate': np.random.normal(2, 0.5, n_dates).cumsum() / 300,  # Inflation rate
        'interest_rate': np.random.normal(2, 0.2, n_dates).cumsum() / 500,  # Interest rate
        'stock_market': np.random.normal(0, 1, n_dates).cumsum() / 50,  # Stock market indicator
        'oil_price': np.random.normal(60, 10, n_dates).cumsum() / 200,  # Oil price
        'bond_yield': np.random.normal(1.5, 0.3, n_dates).cumsum() / 400,  # 10-year Treasury yield
        'geopolitical_risk': np.random.normal(50, 10, n_dates).cumsum() / 300,  # Geopolitical risk index
    })

    # Set date as index for easier merging
    indicators['date'] = pd.to_datetime(indicators['date'])

    return indicators


def connect_to_mt5(login, password, server="MetaQuotes-Demo"):
    """
    Connect to the MetaTrader 5 terminal
    """
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return False

    # Connect to account
    authorized = mt5.login(login, password, server)
    if not authorized:
        print(f"Failed to connect to account {login}, error code: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"Connected to account {login}")
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
        logger.error(f"Failed to get historical data: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Handle different volume column names
    if 'volume' not in df.columns:
        if 'tick_volume' in df.columns:
            logger.info("Using 'tick_volume' for 'volume'")
            df['volume'] = df['tick_volume']
        elif 'real_volume' in df.columns:
            logger.info("Using 'real_volume' for 'volume'")
            df['volume'] = df['real_volume']
        else:
            logger.info("No volume data found, creating placeholder")
            df['volume'] = 1.0

    # Convert to Arizona time
    df['arizona_time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(ARIZONA_TZ)

    logger.info(f"Fetched {len(df)} historical data points")
    return df


def filter_5pm_data(df):
    """
    Filter data to only include rows at 5 PM Arizona time
    """
    df['hour'] = df['arizona_time'].dt.hour
    filtered_df = df[df['hour'] == TARGET_HOUR].copy()
    logger.info(f"Filtered to {len(filtered_df)} data points at 5 PM Arizona time")
    return filtered_df


def add_datetime_features(df):
    """
    Add cyclical datetime features with expanded features for Wednesday
    """
    # Extract datetime components
    df['day_of_week'] = df['arizona_time'].dt.dayofweek
    df['day_of_month'] = df['arizona_time'].dt.day
    df['day_of_year'] = df['arizona_time'].dt.dayofyear
    df['month'] = df['arizona_time'].dt.month
    df['quarter'] = df['arizona_time'].dt.quarter
    df['year'] = df['arizona_time'].dt.year
    df['week_of_year'] = df['arizona_time'].dt.isocalendar().week

    # Create cyclical features for time-based variables to capture their circular nature
    # Sine and cosine transformations for days of week (0-6)
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))

    # Sine and cosine transformations for months (1-12)
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))

    # Sine and cosine transformations for days of month (1-31)
    df['day_of_month_sin'] = np.sin((df['day_of_month'] - 1) * (2 * np.pi / 31))
    df['day_of_month_cos'] = np.cos((df['day_of_month'] - 1) * (2 * np.pi / 31))

    # Day of year (1-366)
    df['day_of_year_sin'] = np.sin((df['day_of_year'] - 1) * (2 * np.pi / 366))
    df['day_of_year_cos'] = np.cos((df['day_of_year'] - 1) * (2 * np.pi / 366))

    # Is weekend feature (binary)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Is month end/start features
    df['is_month_start'] = df['day_of_month'].apply(lambda x: 1 if x <= 3 else 0)
    df['is_month_end'] = df['day_of_month'].apply(lambda x: 1 if x >= 28 else 0)

    # Quarter features
    df['quarter_sin'] = np.sin((df['quarter'] - 1) * (2 * np.pi / 4))
    df['quarter_cos'] = np.cos((df['quarter'] - 1) * (2 * np.pi / 4))

    # Add day of week one-hot encoding (some models perform better with this)
    for i in range(7):
        df[f'day_{i}'] = (df['day_of_week'] == i).astype(int)

    # Add lunar cycle phase (can affect trading psychology)
    # This is a simplified approximation
    days_since_new_moon = (df['day_of_year'] % 29.53).astype(int)
    df['lunar_sin'] = np.sin(days_since_new_moon * (2 * np.pi / 29.53))
    df['lunar_cos'] = np.cos(days_since_new_moon * (2 * np.pi / 29.53))

    # Add holiday proximity indicators (distance to major holidays that affect markets)
    # Simplified approach - in real implementation, use actual holiday calendars
    df['days_to_end_of_month'] = df['arizona_time'].dt.days_in_month - df['day_of_month']

    # NEW: Special features for Wednesdays (which showed poor performance)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)

    # NEW: Market correlation features
    df['is_first_half_week'] = (df['day_of_week'] < 3).astype(int)
    df['is_second_half_week'] = (df['day_of_week'] >= 3).astype(int)

    # NEW: Special features for beginning/middle/end of month
    df['is_early_month'] = (df['day_of_month'] <= 10).astype(int)
    df['is_mid_month'] = ((df['day_of_month'] > 10) & (df['day_of_month'] <= 20)).astype(int)
    df['is_late_month'] = (df['day_of_month'] > 20).astype(int)

    return df


def detect_market_regime(df, window=20):
    """
    Detect market regimes (trending, mean-reverting, volatile) with improved detection
    Based on research paper findings on regime-based trading
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std()

    # Add realized volatility measures (standard, high frequency)
    # FIX: Corrected rolling volatility calculations
    df['realized_vol_10'] = df['returns'].rolling(window=10).apply(lambda x: np.sqrt(np.sum(x**2)) * np.sqrt(252))
    df['realized_vol_20'] = df['returns'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)) * np.sqrt(252))

    # Parkinson volatility estimator (uses high-low range)
    df['high_low_ratio'] = df['high'] / df['low']
    df['log_high_low'] = np.log(df['high_low_ratio'])
    # FIX: Corrected Parkinson volatility calculation
    df['parkinsons_vol'] = df['log_high_low'].rolling(window=window).apply(
        lambda x: np.sqrt((1 / (4 * np.log(2))) * np.sum(x**2) / window)
    )

    # Calculate skewness and kurtosis of returns
    df['returns_skew'] = df['returns'].rolling(window=window).apply(lambda x: skew(x))
    df['returns_kurt'] = df['returns'].rolling(window=window).apply(lambda x: kurtosis(x))

    # Calculate autocorrelation - negative values suggest mean reversion
    df['autocorrelation'] = df['returns'].rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x.dropna()) > 1 else np.nan
    )

    # Calculate trend strength using Hurst exponent approximation
    def hurst_exponent(returns, lags=range(2, 20)):
        tau = []
        std = []
        if len(returns.dropna()) < max(lags) + 1:
            return np.nan

        for lag in lags:
            # Construct a new series with lagged returns
            series_lagged = pd.Series(returns).diff(lag).dropna()
            if len(series_lagged) < 2:  # Need at least 2 points for std
                continue
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
        lambda x: hurst_exponent(x) if len(x.dropna()) > window else np.nan
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

    # Create a composite regime indicator (could be used for model switching)
    df['regime'] = 0  # Default/normal
    df.loc[df['regime_trending'] == 1, 'regime'] = 1  # Trending
    df.loc[df['regime_mean_reverting'] == 1, 'regime'] = 2  # Mean-reverting
    df.loc[df['regime_volatile'] == 1, 'regime'] = 3  # Volatile

    # NEW: Trend strength indicators
    df['adx_trend'] = df['adx'] > 25 if 'adx' in df.columns else np.nan

    # NEW: Volatility regime timing
    df['vol_expansion'] = (df['volatility'] > df['volatility'].shift(1)).astype(int)
    df['vol_contraction'] = (df['volatility'] < df['volatility'].shift(1)).astype(int)

    # NEW: Rate of change of volatility
    df['vol_roc'] = df['volatility'].pct_change() * 100

    # NEW: Volatility of volatility
    df['vol_of_vol'] = df['vol_roc'].rolling(window=window).std()

    return df


def add_wavelet_features(df, column='close', scales=[2, 4, 8, 16]):
    """
    Add wavelet transformation features
    Based on research paper findings on wavelet decomposition for better feature extraction
    Requires PyWavelets package (pip install PyWavelets)
    """
    try:
        import pywt

        # Get the data we want to transform
        data = df[column].values

        for scale in scales:
            # Calculate wavelet coefficients
            coeff, _ = pywt.dwt(data, 'haar')

            # Add as features
            df[f'wavelet_{column}_{scale}'] = np.nan
            df[f'wavelet_{column}_{scale}'].iloc[scale:] = coeff

            # Fill NaN values with forward fill then backward fill
            df[f'wavelet_{column}_{scale}'] = df[f'wavelet_{column}_{scale}'].fillna(method='ffill').fillna(
                method='bfill')

    except ImportError:
        logger.warning("PyWavelets not installed, skipping wavelet features")

    return df


def make_json_serializable(obj):
    """
    Convert NumPy types to Python types for JSON serialization
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj


def add_technical_indicators(df):
    """
    Add technical analysis indicators using pandas - enhanced version
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

        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        # Hull Moving Average (a more responsive moving average)
        for window in [9, 16, 25]:
            # Step 1: Calculate WMA with period n/2
            half_length = int(window / 2)
            df[f'wma_half_{window}'] = df['close'].rolling(window=half_length).apply(
                lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                raw=True
            )

            # Step 2: Calculate WMA for period n/4
            quarter_length = int(window / 4)
            df[f'wma_quarter_{window}'] = df['close'].rolling(window=quarter_length).apply(
                lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                raw=True
            )

            # Step 3: Calculate HMA
            df[f'hma_{window}'] = df[f'wma_quarter_{window}'].rolling(window=int(np.sqrt(window))).apply(
                lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                raw=True
            )

            # Clean up intermediate columns
            df = df.drop(columns=[f'wma_half_{window}', f'wma_quarter_{window}'])

        # Price relative to moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']

        # Moving average crossovers - important trading signals
        df['sma_5_10_cross'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)  # Golden/Death cross
        df['ema_5_10_cross'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)
        df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)  # EMA-based golden/death cross

        # Price momentum and acceleration
        for window in [3, 5, 10, 20]:
            # Momentum
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            # Rate of change (percentage momentum)
            df[f'roc_{window}'] = ((df['close'] / df['close'].shift(window)) - 1) * 100

        # Acceleration (second derivative of price)
        df['acceleration'] = df['close_diff'].diff()
        df['acceleration_pct'] = df['close_diff_pct'].diff()

        # Volatility indicators
        df['volatility_10'] = df['close_diff_pct'].rolling(window=10).std()
        df['volatility_20'] = df['close_diff_pct'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['rsi_14'] = calculate_rsi(df['close'], window=14)
        df['rsi_7'] = calculate_rsi(df['close'], window=7)
        df['rsi_21'] = calculate_rsi(df['close'], window=21)

        # RSI extreme levels and divergences
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

        # RSI divergence (price making higher high while RSI makes lower high)
        df['price_higher_high'] = ((df['high'] > df['high'].shift(1)) &
                                   (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['rsi_lower_high'] = ((df['rsi_14'] < df['rsi_14'].shift(1)) &
                                (df['rsi_14'].shift(1) > df['rsi_14'].shift(2))).astype(int)
        df['bearish_divergence'] = ((df['price_higher_high'] == 1) &
                                    (df['rsi_lower_high'] == 1)).astype(int)

        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            middle_band = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            bb_width = (upper_band - lower_band) / middle_band
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            return upper_band, middle_band, lower_band, bb_width, bb_position

        upper_band, middle_band, lower_band, bb_width, bb_position = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper_band
        df['bb_middle'] = middle_band
        df['bb_lower'] = lower_band
        df['bb_width'] = bb_width
        df['bb_position'] = bb_position

        # MACD (Moving Average Convergence Divergence)
        def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        macd_line, signal_line, histogram = calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # MACD crossovers (important trading signals)
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1,
                                        np.where(df['macd'] < df['macd_signal'], -1, 0))

        # MACD histogram sign change
        df['macd_hist_sign_change'] = np.where(
            (np.sign(df['macd_hist']) != np.sign(df['macd_hist'].shift(1))), 1, 0)

        # Stochastic Oscillator
        def calculate_stochastic(high, low, close, k_period=14, d_period=3):
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=d_period).mean()
            return k, d

        k, d = calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k
        df['stoch_d'] = d

        # Stochastic crossover
        df['stoch_crossover'] = np.where(df['stoch_k'] > df['stoch_d'], 1,
                                         np.where(df['stoch_k'] < df['stoch_d'], -1, 0))

        # Stochastic overbought/oversold
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

        # Average True Range (ATR)
        def calculate_atr(high, low, close, window=14):
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(window=window).mean()

            return atr

        df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'])
        df['atrp_14'] = (df['atr_14'] / df['close']) * 100  # ATR as percentage of price

        # Money Flow Index (MFI)
        def calculate_mfi(high, low, close, volume, window=14):
            tp = (high + low + close) / 3
            raw_money_flow = tp * volume
            money_flow_positive = np.where(tp > tp.shift(1), raw_money_flow, 0)
            money_flow_negative = np.where(tp < tp.shift(1), raw_money_flow, 0)
            money_flow_positive = pd.Series(money_flow_positive, index=high.index)
            money_flow_negative = pd.Series(money_flow_negative, index=high.index)
            positive_sum = money_flow_positive.rolling(window=window).sum()
            negative_sum = money_flow_negative.rolling(window=window).sum()
            money_flow_ratio = positive_sum / negative_sum
            mfi = 100 - (100 / (1 + money_flow_ratio))
            return mfi

        df['mfi_14'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])

        # MFI overbought/oversold
        df['mfi_overbought'] = (df['mfi_14'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi_14'] < 20).astype(int)

        # Commodity Channel Index (CCI)
        def calculate_cci(high, low, close, window=20):
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mean_deviation = abs(tp - sma_tp).rolling(window=window).mean()
            cci = (tp - sma_tp) / (0.015 * mean_deviation)
            return cci

        df['cci_20'] = calculate_cci(df['high'], df['low'], df['close'])

        # Rate of Change (ROC)
        def calculate_roc(close, window=10):
            return ((close / close.shift(window)) - 1) * 100

        df['roc_10'] = calculate_roc(df['close'])

        # On Balance Volume (OBV)
        def calculate_obv(close, volume):
            obv = pd.Series(0, index=close.index)

            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

            return obv

        df['obv'] = calculate_obv(df['close'], df['volume'])

        # OBV rate of change
        df['obv_roc_10'] = calculate_roc(df['obv'], 10)

        # Volume indicators
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']

        # Volume change
        df['volume_change'] = df['volume'].pct_change() * 100

        # Price-volume divergence
        df['price_up_volume_down'] = ((df['close_diff'] > 0) & (df['volume_change'] < 0)).astype(int)
        df['price_down_volume_up'] = ((df['close_diff'] < 0) & (df['volume_change'] > 0)).astype(int)

        # Z-score of price
        df['zscore_20'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

        # Advanced indicators

        # Ichimoku Cloud
        def calculate_ichimoku(high, low, close):
            # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
            tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2

            # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
            kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2

            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted 26 periods forward
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods, shifted 26 periods forward
            senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)

            # Chikou Span (Lagging Span): Close price shifted 26 periods backward
            chikou_span = close.shift(-26)

            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

        tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df['high'], df['low'], df['close'])
        df['ichimoku_tenkan'] = tenkan
        df['ichimoku_kijun'] = kijun
        df['ichimoku_senkou_a'] = span_a
        df['ichimoku_senkou_b'] = span_b
        df['ichimoku_chikou'] = chikou

        # Ichimoku trading signals
        df['tenkan_kijun_cross'] = np.where(df['ichimoku_tenkan'] > df['ichimoku_kijun'], 1,
                                            np.where(df['ichimoku_tenkan'] < df['ichimoku_kijun'], -1, 0))

        # Price relative to cloud
        df['price_above_cloud'] = np.where((df['close'] > df['ichimoku_senkou_a']) &
                                           (df['close'] > df['ichimoku_senkou_b']), 1, 0)
        df['price_below_cloud'] = np.where((df['close'] < df['ichimoku_senkou_a']) &
                                           (df['close'] < df['ichimoku_senkou_b']), 1, 0)
        df['cloud_bullish'] = np.where(df['ichimoku_senkou_a'] > df['ichimoku_senkou_b'], 1, 0)

        # Elder Force Index
        # Force Index: Price change * Volume
        df['force_index_1'] = df['close'].diff(1) * df['volume']
        df['force_index_13'] = df['force_index_1'].ewm(span=13, adjust=False).mean()

        # Choppiness Index (market is trending or trading sideways)
        def calculate_choppiness_index(high, low, close, window=14):
            atr_sum = calculate_atr(high, low, close, 1).rolling(window=window).sum()
            high_low_range = high.rolling(window=window).max() - low.rolling(window=window).min()
            ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(window)
            return ci

        df['choppiness_14'] = calculate_choppiness_index(df['high'], df['low'], df['close'])

        # ADX (Average Directional Index) - trend strength
        def calculate_adx(high, low, close, window=14):
            # Calculate +DM and -DM
            high_change = high.diff()
            low_change = low.diff()

            # +DM
            pos_dm = np.where((high_change > 0) & (high_change > low_change.abs()), high_change, 0)
            pos_dm = pd.Series(pos_dm, index=high.index)

            # -DM
            neg_dm = np.where((low_change < 0) & (low_change.abs() > high_change), low_change.abs(), 0)
            neg_dm = pd.Series(neg_dm, index=low.index)

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate smoothed values
            smoothed_tr = tr.rolling(window=window).sum()
            smoothed_pos_dm = pos_dm.rolling(window=window).sum()
            smoothed_neg_dm = neg_dm.rolling(window=window).sum()

            # Calculate +DI and -DI
            pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
            neg_di = 100 * (smoothed_neg_dm / smoothed_tr)

            # Calculate DX and ADX
            dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
            adx = dx.rolling(window=window).mean()

            return pos_di, neg_di, adx

        pos_di, neg_di, adx = calculate_adx(df['high'], df['low'], df['close'])
        df['adx_pos_di'] = pos_di
        df['adx_neg_di'] = neg_di
        df['adx'] = adx

        # ADX trend signals
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
        df['adx_trend_direction'] = np.where(df['adx_pos_di'] > df['adx_neg_di'], 1, -1)

        # Williams %R
        def calculate_williams_r(high, low, close, window=14):
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return wr

        df['williams_r_14'] = calculate_williams_r(df['high'], df['low'], df['close'])

        # Williams %R overbought/oversold
        df['williams_r_overbought'] = (df['williams_r_14'] > -20).astype(int)
        df['williams_r_oversold'] = (df['williams_r_14'] < -80).astype(int)

        # NEW INDICATORS

        # Average Directional Movement Index Rating (ADXR)
        df['adxr'] = (df['adx'] + df['adx'].shift(14)) / 2

        # Linear Regression Slope
        def calc_linreg_slope(series, window=20):
            slopes = []
            for i in range(len(series) - window + 1):
                y = series.iloc[i:i + window].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            return pd.Series(slopes, index=series.index[window - 1:])

        df['price_slope_20'] = calc_linreg_slope(df['close'], 20)
        df.loc[:19, 'price_slope_20'] = df['price_slope_20'].iloc[0]  # Fill initial NaNs

        # Dynamic RSI Levels (adaptive RSI thresholds based on volatility)
        df['rsi_high_threshold'] = 70 + (df['volatility_20'] * 50)  # More volatile = wider thresholds
        df['rsi_low_threshold'] = 30 - (df['volatility_20'] * 50)

        # Volume-weighted RSI
        df['vol_weight'] = df['volume'] / df['volume'].rolling(14).mean()
        df['vol_weighted_rsi'] = df['rsi_14'] * df['vol_weight']

        # Focus on key levels - distance from round numbers/psychologically important levels
        key_levels = [1000, 1500, 1600, 1700, 1800, 1900, 2000, 2100]  # Common gold price levels
        df['distance_to_key_level'] = df['close'].apply(
            lambda price: min([abs(price - level) / price for level in key_levels])
        )

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return df


def handle_outliers(df, columns, method='winsorize', threshold=3.0):
    """Apply outlier handling to specific columns"""
    logger.info(f"Handling outliers using {method} method with threshold {threshold}")
    df_clean = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'winsorize':
            # Winsorization (capping)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore':
            # Z-score filtering
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < threshold)
            df_clean.loc[~filtered_entries, col] = np.nan

        elif method == 'isolation_forest':
            # Isolation Forest for anomaly detection
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            yhat = iso.fit_predict(df[[col]])
            mask = yhat != -1
            df_clean.loc[~mask, col] = np.nan

    # Fill NaNs with median or forward fill
    for col in columns:
        if col in df.columns:
            if df_clean[col].isna().any():
                if len(df_clean[col].dropna()) > 0:
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(df_clean[col].median())

    return df_clean


def add_macroeconomic_data(df, macro_df):
    """
    Add macroeconomic data relevant to gold prices
    In real implementation, use actual economic data
    """
    logger.info("Adding macroeconomic indicators")

    # Convert dates for merging
    df['date'] = pd.to_datetime(df['arizona_time'].dt.date)

    # Merge with macro data
    df = pd.merge(df, macro_df, on='date', how='left')

    # Forward fill any missing values
    econ_cols = ['usd_index', 'inflation_rate', 'interest_rate',
                 'stock_market', 'oil_price', 'bond_yield', 'geopolitical_risk']

    for col in econ_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')

    # Calculate rates of change for economic indicators
    for col in econ_cols:
        if col in df.columns:
            df[f'{col}_roc'] = df[col].pct_change() * 100

    # Add interaction features between gold and economic factors
    if 'usd_index' in df.columns:
        df['gold_usd_ratio'] = df['close'] / df['usd_index']

    if 'oil_price' in df.columns:
        df['gold_oil_ratio'] = df['close'] / df['oil_price']

    if 'inflation_rate' in df.columns:
        df['gold_inflation_adjusted'] = df['close'] / (1 + df['inflation_rate'] / 100)

    # Add economic surprise indicators
    if 'inflation_rate' in df.columns:
        df['inflation_surprise'] = df['inflation_rate'] - df['inflation_rate'].shift(20)

    if 'interest_rate' in df.columns:
        df['interest_rate_change'] = df['interest_rate'] - df['interest_rate'].shift(5)
        df['gold_interest_ratio'] = df['close'] / (1 + df['interest_rate'] / 100)

    return df


def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """
    Add lagged features for selected columns with more lags as suggested in research
    """
    # Key indicators to lag - focus on the most important ones
    key_indicators = [
        'close', 'close_diff_pct', 'rsi_14', 'macd', 'bb_position',
        'volatility_20', 'adx', 'adx_trend_direction', 'stoch_k', 'mfi_14',
        'volume_change', 'price_slope_20'
    ]

    # Add special indicators for Wednesday
    if 'is_wednesday' in df.columns and df['is_wednesday'].sum() > 0:
        key_indicators.extend(['gold_usd_ratio', 'gold_oil_ratio', 'obv', 'atr_14'])

    # Add lags for all key indicators
    for col in key_indicators:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Add rate of change between lags for close price
    for lag in lags[1:]:  # Skip the first lag
        if f'close_lag_{lag}' in df.columns and f'close_lag_1' in df.columns:
            df[f'close_lag_{lag}_1_diff'] = ((df[f'close_lag_1'] - df[f'close_lag_{lag}']) /
                                             df[f'close_lag_{lag}']) * 100

    # Add day-specific interaction features
    for day in range(7):
        if f'day_{day}' in df.columns and df[f'day_{day}'].sum() > 0:
            # Create day-specific features for the most important indicators
            for col in ['close_diff_pct', 'rsi_14', 'adx']:
                if col in df.columns:
                    df[f'{col}_day_{day}'] = df[col] * df[f'day_{day}']

    return df


def add_target_variables(df):
    """
    Add target variables for prediction including multiple timeframes
    """
    # Calculate standard next period close price change
    df['next_close'] = df['close'].shift(-1)
    df['next_close_change_pct'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Add directional target (binary classification)
    df['next_direction'] = np.where(df['next_close'] > df['close'], 1, 0)

    # Add multi-step targets for future periods
    for i in range(2, 6):
        df[f'close_future_{i}'] = df['close'].shift(-i)
        df[f'change_future_{i}_pct'] = ((df[f'close_future_{i}'] - df['close']) / df['close']) * 100

    # Add volatility target (predict future volatility)
    future_std = df['close_diff_pct'].rolling(window=5).std().shift(-5)
    df['future_volatility'] = future_std

    # Add probability of extreme move target
    # Define extreme move as > 2 standard deviations
    vol_threshold = df['volatility_20'] * 2
    extreme_moves = []

    for i in range(len(df) - 5):
        max_move = df['close_diff_pct'].iloc[i + 1:i + 6].abs().max()
        extreme_moves.append(1 if max_move > vol_threshold.iloc[i] else 0)

    extreme_moves.extend([np.nan] * 5)  # Pad the end
    df['extreme_move_5d'] = extreme_moves

    # Add target for regime switches (if market regime detection is enabled)
    if 'regime' in df.columns:
        df['regime_switch'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['next_regime'] = df['regime'].shift(-1)

    # NEW: Add magnitude-weighted direction target
    # This weights the target by the size of the move, emphasizing larger moves
    df['weighted_direction'] = df['next_close_change_pct'] * np.sign(df['next_close_change_pct'])

    # NEW: Add compound multi-period return
    df['compound_3day_return'] = ((1 + df['next_close_change_pct'] / 100) *
                                  (1 + df['change_future_2_pct'] / 100) *
                                  (1 + df['change_future_3_pct'] / 100) - 1) * 100

    # NEW: Add range-based target (high-low range for next period)
    if 'high' in df.columns and 'low' in df.columns:
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        df['next_range_pct'] = ((df['next_high'] - df['next_low']) / df['close']) * 100

    return df


def prepare_features_and_targets(df, target_col='next_close_change_pct', feature_blacklist=None,
                                 handle_outliers_cols=None):
    """
    Prepare features and target variables with improved handling of missing values and outliers
    """
    # Default blacklist if none provided
    if feature_blacklist is None:
        feature_blacklist = [
            'time', 'arizona_time', 'date', 'next_close', 'hour',
            'close_future_2', 'close_future_3', 'close_future_4', 'close_future_5',
            'next_high', 'next_low'
        ]

    # Handle outliers in specific columns if requested
    if HANDLE_OUTLIERS and handle_outliers_cols is not None:
        # Default to target column if not specified
        if not handle_outliers_cols:
            handle_outliers_cols = [target_col]

        # Handle outliers in the selected columns
        df = handle_outliers(df, handle_outliers_cols, method='winsorize', threshold=3.0)

    # Drop unnecessary columns
    drop_cols = feature_blacklist + [f'close_future_{i}' for i in range(2, 6)]
    feature_df = df.drop(columns=drop_cols, errors='ignore')

    # Handle NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # For features, forward-fill then backward-fill
    feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')

    # Get remaining NaN columns and fill with zeros
    nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Filling these columns with zeros: {nan_cols}")
        feature_df[nan_cols] = feature_df[nan_cols].fillna(0)

    # Separate features and target
    y = feature_df[target_col].values

    # Remove target columns from features
    target_cols = ['next_close_change_pct', 'next_direction', 'future_volatility', 'extreme_move_5d',
                   'regime_switch', 'next_regime', 'weighted_direction', 'compound_3day_return',
                   'next_range_pct']
    target_cols += [f'change_future_{i}_pct' for i in range(2, 6)]

    # Make sure all target columns exist before dropping
    target_cols = [col for col in target_cols if col in feature_df.columns]

    X = feature_df.drop(columns=target_cols, errors='ignore').values
    feature_names = feature_df.drop(columns=target_cols, errors='ignore').columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Check for class imbalance in directional prediction
    if 'next_direction' in feature_df.columns:
        up_pct = feature_df['next_direction'].mean() * 100
        logger.info(f"Class balance - Up: {up_pct:.1f}%, Down: {100 - up_pct:.1f}%")

    return X, y, feature_names


def select_features(X, y, feature_names, method='rfe', n_features=50):
    """
    Perform feature selection to reduce noise and focus on important features
    """
    logger.info(f"Selecting top {n_features} features using {method} method")

    # Make sure n_features is not greater than the number of features available
    n_features = min(n_features, X.shape[1])

    # Initialize variable to store selected feature indices
    selected_indices = None
    selected_feature_names = None

    if method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=0.2)
        selector.fit(X, y)
        selected_indices = np.where(selector.support_)[0]

    elif method == 'random_forest':
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        selected_indices = np.argsort(importances)[-n_features:]

    elif method == 'correlation':
        # Correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))
        selected_indices = np.argsort(correlations)[-n_features:]

    elif method == 'mutual_info':
        # Mutual information regression
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X, y)
        selected_indices = np.argsort(mi_scores)[-n_features:]

    else:
        logger.warning(f"Unknown feature selection method: {method}, using all features")
        selected_indices = np.arange(X.shape[1])

    # Get selected feature names
    if selected_indices is not None and feature_names is not None:
        selected_feature_names = [feature_names[i] for i in selected_indices]
        logger.info(f"Selected features: {selected_feature_names[:10]}...")

    # Return selected features
    X_selected = X[:, selected_indices]

    return X_selected, selected_indices, selected_feature_names


def scale_features(X_train, X_val=None, X_test=None, scaler_type='robust'):
    """
    Scale features using specified scaler type
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown scaler type {scaler_type}, using RobustScaler")
        scaler = RobustScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and test sets if provided
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM/GRU models with proper shape checking
    """
    X_seq, y_seq = [], []

    # Ensure X is 2D before processing
    if len(X.shape) != 2:
        raise ValueError(f"Expected X to be 2D array with shape (samples, features), got shape {X.shape}")

    # Create sequences
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    X_seq_array = np.array(X_seq)
    y_seq_array = np.array(y_seq)

    # Validate output shapes
    if len(X_seq_array.shape) != 3:
        raise ValueError(
            f"Expected X_seq to be 3D with shape (samples, lookback, features), got shape {X_seq_array.shape}")

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
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Training set: {len(train_df)} samples ({train_size * 100:.1f}%)")
    logger.info(f"Validation set: {len(val_df)} samples ({val_size * 100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} samples ({(1 - train_size - val_size) * 100:.1f}%)")

    # Log date ranges
    logger.info(f"Train period: {train_df['arizona_time'].min()} to {train_df['arizona_time'].max()}")
    logger.info(f"Validation period: {val_df['arizona_time'].min()} to {val_df['arizona_time'].max()}")
    logger.info(f"Test period: {test_df['arizona_time'].min()} to {test_df['arizona_time'].max()}")

    return train_df, val_df, test_df


def r2_keras(y_true, y_pred):
    """
    Custom R² metric for Keras
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (same sign)
    """
    return K.mean(K.cast(K.sign(y_true) == K.sign(y_pred), 'float32'))


def directional_weighted_loss(alpha=0.5):
    """
    Custom loss function combining MSE with directional accuracy
    Puts extra weight on getting the direction of larger moves correct
    """

    def loss(y_true, y_pred):
        # MSE component
        mse = K.mean(K.square(y_true - y_pred))

        # Directional component - penalize wrong directions
        dir_true = K.sign(y_true)
        dir_pred = K.sign(y_pred)
        dir_match = K.cast(K.equal(dir_true, dir_pred), 'float32')
        dir_penalty = 1.0 - dir_match

        # Weight by size of true value (larger moves should be predicted better)
        move_size_weight = K.abs(y_true) / (K.mean(K.abs(y_true)) + K.epsilon())
        weighted_dir_penalty = move_size_weight * dir_penalty

        # Combined loss
        return (1.0 - alpha) * mse + alpha * K.mean(weighted_dir_penalty)

    return loss


def directional_accuracy_numpy(y_true, y_pred):
    """Calculate directional accuracy for numpy arrays"""
    return np.mean((np.sign(y_true) == np.sign(y_pred)).astype(int))


def build_simple_nn_model(input_shape, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a simple neural network for comparison with complex models
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # Use custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_attention_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build LSTM model with attention mechanism
    """
    # Determine number of units based on complexity
    if complexity == 'low':
        units = [64, 48, 32]
    elif complexity == 'medium':
        units = [128, 64, 32]
    elif complexity == 'high':
        units = [256, 128, 64, 32]
    else:
        units = [128, 64, 32]  # Default to medium

    # Input layer
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

    # Attention layer
    attention_layer = Attention()([drop_2, drop_2])

    # Third LSTM layer
    lstm_3 = LSTM(units[2],
                  recurrent_dropout=dropout_rate,
                  recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(attention_layer)
    norm_3 = BatchNormalization()(lstm_3)
    drop_3 = Dropout(dropout_rate)(norm_3)

    # Dense layers
    dense_1 = Dense(max(16, units[2] // 2), activation='relu',
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5))(drop_3)
    norm_4 = BatchNormalization()(dense_1)
    drop_4 = Dropout(dropout_rate)(norm_4)

    # Output layer
    output_layer = Dense(1)(drop_4)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile with custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_cnn_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build hybrid CNN-LSTM model
    """
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

    # Input layer
    input_layer = Input(shape=input_shape)

    # CNN layers
    conv_1 = Conv1D(filters=conv_filters[0], kernel_size=3, padding='same', activation='relu')(input_layer)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = MaxPooling1D(pool_size=2, padding='same')(conv_1)
    conv_1 = Dropout(dropout_rate)(conv_1)

    conv_2 = Conv1D(filters=conv_filters[1], kernel_size=3, padding='same', activation='relu')(conv_1)
    conv_2 = BatchNormalization()(conv_2)

    if len(conv_filters) > 2:
        conv_2 = MaxPooling1D(pool_size=2, padding='same')(conv_2)
        conv_2 = Dropout(dropout_rate)(conv_2)

        conv_3 = Conv1D(filters=conv_filters[2], kernel_size=3, padding='same', activation='relu')(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        conv_output = Dropout(dropout_rate)(conv_3)
    else:
        conv_output = Dropout(dropout_rate)(conv_2)

    # LSTM layers
    lstm_1 = LSTM(lstm_units[0], return_sequences=(len(lstm_units) > 1),
                  recurrent_dropout=dropout_rate,
                  recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5))(conv_output)
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

    # Use custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build LSTM model with variable complexity
    """
    if complexity == 'low':
        units = [64, 32]
    elif complexity == 'medium':
        units = [128, 64, 32]
    elif complexity == 'high':
        units = [256, 128, 64, 32]
    else:
        units = [128, 64, 32]  # Default to medium

    model = Sequential()

    # First LSTM layer with return sequences
    model.add(LSTM(
        units[0],
        return_sequences=True if len(units) > 1 else False,
        input_shape=input_shape,
        recurrent_dropout=dropout_rate,
        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers
    for i in range(1, len(units) - 1):
        model.add(LSTM(
            units[i],
            return_sequences=True,
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Last LSTM layer (if more than one)
    if len(units) > 1:
        model.add(LSTM(
            units[-1],
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Dense(max(16, units[-1] // 2), activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Use custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_gru_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build GRU model with variable complexity
    """
    if complexity == 'low':
        units = [64, 32]
    elif complexity == 'medium':
        units = [128, 64, 32]
    elif complexity == 'high':
        units = [256, 128, 64, 32]
    else:
        units = [128, 64, 32]  # Default to medium

    model = Sequential()

    # First GRU layer with return sequences
    model.add(GRU(
        units[0],
        return_sequences=True if len(units) > 1 else False,
        input_shape=input_shape,
        recurrent_dropout=dropout_rate,
        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional GRU layers
    for i in range(1, len(units) - 1):
        model.add(GRU(
            units[i],
            return_sequences=True,
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Last GRU layer (if more than one)
    if len(units) > 1:
        model.add(GRU(
            units[-1],
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Dense(max(16, units[-1] // 2), activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Use custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_bidirectional_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build bidirectional LSTM model
    """
    if complexity == 'low':
        units = [64, 32]
    elif complexity == 'medium':
        units = [128, 64, 32]
    elif complexity == 'high':
        units = [256, 128, 64, 32]
    else:
        units = [128, 64, 32]  # Default to medium

    model = Sequential()

    # First bidirectional LSTM layer
    model.add(Bidirectional(LSTM(
        units[0],
        return_sequences=True if len(units) > 1 else False,
        recurrent_dropout=dropout_rate,
        recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
    ), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Additional bidirectional LSTM layers
    for i in range(1, len(units) - 1):
        model.add(Bidirectional(LSTM(
            units[i],
            return_sequences=True,
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        )))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Last bidirectional LSTM layer (if more than one)
    if len(units) > 1:
        model.add(Bidirectional(LSTM(
            units[-1],
            recurrent_dropout=dropout_rate,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        )))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Dense(max(16, units[-1] // 2), activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Use custom loss if needed
    if WEIGHTED_LOSS:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=directional_weighted_loss(0.3),
            metrics=['mae', directional_accuracy, r2_keras]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', directional_accuracy, r2_keras]
        )

    return model


def build_traditional_model(model_type='random_forest', params=None):
    """
    Build a traditional ML model (Random Forest, GBM, Extra Trees)
    """
    # Default parameters
    if params is None:
        params = {}

    # Build model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            random_state=42
        )
    elif model_type == 'extra_trees':
        model = ExtraTreesRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            random_state=42,
            n_jobs=-1
        )
    else:
        logger.warning(f"Unknown traditional model type: {model_type}, using RandomForest")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    return model


def build_model_by_type(model_type, input_shape=None, params=None):
    """
    Factory function to build models based on type
    """
    # Default parameters
    if params is None:
        params = {}

    dropout_rate = params.get('dropout_rate', 0.3)
    learning_rate = params.get('learning_rate', 0.001)
    complexity = params.get('complexity', 'medium')

    # Traditional models don't need input_shape
    if model_type in ['random_forest', 'gbm', 'extra_trees']:
        return build_traditional_model(model_type, params)

    # Neural network models
    if input_shape is None:
        raise ValueError("input_shape is required for neural network models")

    if model_type == 'simple_nn':
        return build_simple_nn_model(input_shape, dropout_rate, learning_rate)
    elif model_type == 'lstm':
        return build_lstm_model(input_shape, complexity, dropout_rate, learning_rate)
    elif model_type == 'gru':
        return build_gru_model(input_shape, complexity, dropout_rate, learning_rate)
    elif model_type == 'bidirectional':
        return build_bidirectional_model(input_shape, complexity, dropout_rate, learning_rate)
    elif model_type == 'cnn_lstm':
        return build_cnn_lstm_model(input_shape, complexity, dropout_rate, learning_rate)
    elif model_type == 'attention':
        return build_attention_lstm_model(input_shape, complexity, dropout_rate, learning_rate)
    else:
        logger.warning(f"Unknown model type {model_type}, using LSTM")
        return build_lstm_model(input_shape, complexity, dropout_rate, learning_rate)


def hyperparameter_grid_search(X_train, y_train, X_val, y_val, model_type='lstm', is_sequence=True):
    """
    Enhanced hyperparameter grid search with improved parameter ranges
    """
    # Define hyperparameter grid
    if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention']:
        hyperparams = {
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'complexity': ['low', 'medium', 'high'],
            'batch_size': [16, 32, 64]
        }
    elif model_type == 'simple_nn':
        hyperparams = {
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64]
        }
    else:  # Traditional models
        if model_type == 'random_forest':
            hyperparams = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gbm':
            hyperparams = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
        else:
            hyperparams = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }

    best_val_metric = -float('inf')  # For directional accuracy (higher is better)
    best_params = {}
    best_model = None

    # Log total combinations
    import itertools
    param_combinations = list(itertools.product(*hyperparams.values()))
    total_combinations = len(param_combinations)
    logger.info(f"Starting grid search with {total_combinations} combinations for {model_type}")

    # Try all combinations
    for i, combination in enumerate(param_combinations):
        # Create parameter dictionary
        current_params = dict(zip(hyperparams.keys(), combination))

        logger.info(f"[{i + 1}/{total_combinations}] Training {model_type} with params: {current_params}")

        # Build model
        K.clear_session()

        # Need different handling for sequence models vs traditional
        if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention', 'simple_nn']:
            # Neural network model
            if is_sequence and model_type != 'simple_nn':
                # Build sequence model
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = build_model_by_type(model_type, input_shape, current_params)

                # Train with early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )

                batch_size = current_params.get('batch_size', 32)

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Get directional accuracy
                val_dir_acc = history.history['val_directional_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                logger.info(f"  Validation dir. accuracy: {val_dir_acc:.4f}, loss: {val_loss:.6f}")

                # Use directional accuracy as metric
                val_metric = val_dir_acc

            else:
                # Simple NN model - not sequence based
                input_shape = X_train.shape[1]
                model = build_simple_nn_model(
                    input_shape,
                    dropout_rate=current_params.get('dropout_rate', 0.3),
                    learning_rate=current_params.get('learning_rate', 0.001)
                )

                # Train with early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )

                batch_size = current_params.get('batch_size', 32)

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Get metrics
                val_dir_acc = history.history['val_directional_accuracy'][-1]
                val_loss = history.history['val_loss'][-1]

                logger.info(f"  Validation dir. accuracy: {val_dir_acc:.4f}, loss: {val_loss:.6f}")

                # Use directional accuracy as metric
                val_metric = val_dir_acc

        else:
            # Traditional model
            model = build_model_by_type(model_type, params=current_params)

            # Train model
            model.fit(X_train, y_train)

            # Predict on validation set
            y_pred = model.predict(X_val)

            # Calculate directional accuracy
            val_dir_acc = directional_accuracy_numpy(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)

            logger.info(f"  Validation dir. accuracy: {val_dir_acc:.4f}, MSE: {mse:.6f}")

            # Use directional accuracy as metric
            val_metric = val_dir_acc

        # Check if this model is better
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_params = current_params
            best_model = model
            logger.info(f"  New best model found!")

        # Force garbage collection
        gc.collect()

    logger.info(f"Best hyperparameters for {model_type}: {best_params}")
    logger.info(f"Best validation metric: {best_val_metric:.6f}")

    return best_model, best_params


def optuna_objective(trial, X_train, y_train, X_val, y_val, model_type, is_sequence=True):
    """Objective function for Optuna hyperparameter optimization"""
    params = {}

    # Common hyperparameters for neural networks
    if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention', 'simple_nn']:
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Model-specific hyperparameters
    if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention']:
        params['complexity'] = trial.suggest_categorical('complexity', ['low', 'medium', 'high'])

    # Traditional model hyperparameters
    if model_type == 'random_forest':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 5, 30)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
    elif model_type == 'gbm':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)

    # Build model based on parameters
    if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and is_sequence:
        # Sequence-based models
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model_by_type(model_type, input_shape, params)

        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )

        # Get validation directional accuracy
        val_dir_acc = history.history['val_directional_accuracy'][-1]
        return val_dir_acc

    elif model_type == 'simple_nn' or (model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and not is_sequence):
        # Non-sequence neural network
        input_shape = X_train.shape[1]
        model = build_simple_nn_model(
            input_shape,
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )

        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )

        # Get validation directional accuracy
        val_dir_acc = history.history['val_directional_accuracy'][-1]
        return val_dir_acc

    else:
        # Traditional ML models
        model = build_model_by_type(model_type, params=params)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate directional accuracy
        val_dir_acc = directional_accuracy_numpy(y_val, y_pred)
        return val_dir_acc

    def optimize_with_optuna(X_train, y_train, X_val, y_val, model_type, is_sequence=True, n_trials=50):
        """Run hyperparameter optimization with Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to grid search")
            return hyperparameter_grid_search(X_train, y_train, X_val, y_val, model_type, is_sequence)

        logger.info(f"Starting Optuna hyperparameter optimization for {model_type} with {n_trials} trials")

        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

        # Create objective function
        objective = lambda trial: optuna_objective(
            trial, X_train, y_train, X_val, y_val, model_type, is_sequence
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best parameters for {model_type}: {best_params}")
        logger.info(f"Best directional accuracy: {best_value:.4f}")

        # Build model with best parameters
        if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and is_sequence:
            input_shape = (X_train.shape[1], X_train.shape[2])
            best_model = build_model_by_type(model_type, input_shape, best_params)
        elif model_type == 'simple_nn' or (
                model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and not is_sequence):
            input_shape = X_train.shape[1]
            best_model = build_simple_nn_model(
                input_shape,
                dropout_rate=best_params.get('dropout_rate', 0.3),
                learning_rate=best_params.get('learning_rate', 0.001)
            )
        else:
            best_model = build_model_by_type(model_type, params=best_params)

        return best_model, best_params

    def build_stacked_ensemble(X_train, y_train, X_val, y_val, base_models=None):
        """
        Build a stacked ensemble with diverse base models
        Uses base models' predictions as features for a meta-learner
        """
        logger.info("Building stacked ensemble model")

        if base_models is None:
            # Create diverse base models
            base_models = []
            # Add traditional models
            base_models.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42)))
            base_models.append(('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42)))
            base_models.append(('et', ExtraTreesRegressor(n_estimators=100, random_state=42)))

            # Add neural network if we have enough data
            if len(X_train) >= 500:
                nn_model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                nn_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
                base_models.append(('nn', nn_model))

        # Train base models and get their predictions
        base_predictions_train = np.zeros((X_train.shape[0], len(base_models)))
        base_predictions_val = np.zeros((X_val.shape[0], len(base_models)))
        trained_models = []

        for i, (name, model) in enumerate(base_models):
            logger.info(f"Training base model: {name}")
            if isinstance(model, (Sequential, Model)):
                # Neural network model
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                base_predictions_train[:, i] = model.predict(X_train).flatten()
                base_predictions_val[:, i] = model.predict(X_val).flatten()
            else:
                # Sklearn model
                model.fit(X_train, y_train)
                base_predictions_train[:, i] = model.predict(X_train)
                base_predictions_val[:, i] = model.predict(X_val)

            trained_models.append((name, model))

            # Evaluate individual model
            if isinstance(model, (Sequential, Model)):
                val_metrics = model.evaluate(X_val, y_val, verbose=0)
                val_loss = val_metrics[0]
                val_dir_acc = val_metrics[2]  # directional_accuracy is 3rd metric
            else:
                val_pred = model.predict(X_val)
                val_loss = mean_squared_error(y_val, val_pred)
                val_dir_acc = directional_accuracy_numpy(y_val, val_pred)

            logger.info(f"  {name} - Val Loss: {val_loss:.6f}, Dir Acc: {val_dir_acc:.4f}")

        # Add original features to meta-features
        meta_features_train = np.hstack([base_predictions_train, X_train[:, :20]])  # Use top 20 original features
        meta_features_val = np.hstack([base_predictions_val, X_val[:, :20]])

        # Train meta-learner
        logger.info("Training meta-learner")
        meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_learner.fit(meta_features_train, y_train)

        # Evaluate ensemble
        meta_pred_val = meta_learner.predict(meta_features_val)
        ensemble_mse = mean_squared_error(y_val, meta_pred_val)
        ensemble_dir_acc = directional_accuracy_numpy(y_val, meta_pred_val)

        logger.info(f"Stacked Ensemble - Val MSE: {ensemble_mse:.6f}, Dir Acc: {ensemble_dir_acc:.4f}")

        # Store models and metadata
        ensemble = {
            'base_models': trained_models,
            'meta_learner': meta_learner,
            'performance': {
                'mse': ensemble_mse,
                'directional_accuracy': ensemble_dir_acc
            }
        }

        return ensemble

    def predict_with_stacked_ensemble(ensemble, X):
        """Make predictions with the stacked ensemble"""
        base_models = ensemble['base_models']
        meta_learner = ensemble['meta_learner']

        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(base_models)))
        for i, (name, model) in enumerate(base_models):
            if isinstance(model, (Sequential, Model)):
                base_predictions[:, i] = model.predict(X).flatten()
            else:
                base_predictions[:, i] = model.predict(X)

        # Combine with top 20 original features for meta-learner
        meta_features = np.hstack([base_predictions, X[:, :20]])

        # Make final prediction
        return meta_learner.predict(meta_features)

    def train_day_specific_models(df_5pm):

        """

        Train separate models for each day of the week to address the Wednesday performance issue

        """

        logger.info("Training day-specific models")

        day_models = {}

        day_performance = {}

        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                       6: 'Sunday'}

        # Create a DataFrame to store day performance for visualization

        day_perf_df = pd.DataFrame(columns=['day', 'count', 'mse', 'mae', 'directional_accuracy'])

        for day_idx, day_name in day_mapping.items():

            day_df = df_5pm[df_5pm['day_of_week'] == day_idx].copy()

            if len(day_df) < 30:
                logger.warning(f"Not enough data for {day_name}, skipping")

                continue

            logger.info(f"Training model for {day_name} with {len(day_df)} samples")

            # Split, prepare, and train model for this day

            train_df, val_df, test_df = time_series_split(day_df)

            X_train, y_train, feature_names = prepare_features_and_targets(train_df)

            X_val, y_val, _ = prepare_features_and_targets(val_df)

            X_test, y_test, _ = prepare_features_and_targets(test_df)

            # Use more features for Wednesday (which has poor performance)

            n_features = 80 if day_name == 'Wednesday' else 50

            # Select features specifically for this day

            if USE_FEATURE_SELECTION:

                X_train_selected, selected_indices, selected_features = select_features(

                    X_train, y_train, feature_names, method=FEATURE_SELECTION_METHOD, n_features=n_features

                )

                X_val_selected = X_val[:, selected_indices]

                X_test_selected = X_test[:, selected_indices]

            else:

                X_train_selected = X_train

                X_val_selected = X_val

                X_test_selected = X_test

                selected_features = feature_names

            # Scale features

            X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(

                X_train_selected, X_val_selected, X_test_selected, scaler_type='robust'

            )

            # For Wednesday, use a more robust model or ensemble

            if day_name == 'Wednesday':

                logger.info("Using RandomForest for Wednesday to improve performance")

                model = RandomForestRegressor(n_estimators=200, max_depth=8,

                                              min_samples_leaf=10, random_state=42)

                model.fit(X_train_scaled, y_train)

                # Predict and evaluate

                y_pred = model.predict(X_test_scaled)

                mse = mean_squared_error(y_test, y_pred)

                mae = mean_absolute_error(y_test, y_pred)

                dir_acc = directional_accuracy_numpy(y_test, y_pred)

                day_model = {

                    'model': model,

                    'scaler': scaler,

                    'selected_features': selected_features,

                    'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,

                    'model_type': 'random_forest'

                }

            else:

                # For other days, use LSTM if enough data, otherwise RF

                if len(X_train) >= 100:

                    # Create sequences for LSTM

                    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)

                    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val)

                    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)

                    logger.info(f"Training LSTM for {day_name}")

                    model = build_lstm_model(

                        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),

                        complexity='medium',

                        dropout_rate=0.4  # Higher dropout for regularization

                    )

                    model.fit(

                        X_train_seq, y_train_seq,

                        validation_data=(X_val_seq, y_val_seq),

                        epochs=100,

                        batch_size=32,

                        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],

                        verbose=1

                    )

                    # Predict and evaluate

                    y_pred = model.predict(X_test_seq).flatten()

                    mse = mean_squared_error(y_test_seq, y_pred)

                    mae = mean_absolute_error(y_test_seq, y_pred)

                    dir_acc = directional_accuracy_numpy(y_test_seq, y_pred)

                    day_model = {

                        'model': model,

                        'scaler': scaler,

                        'selected_features': selected_features,

                        'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,

                        'model_type': 'lstm',

                        'is_sequence': True

                    }

                else:

                    logger.info(f"Using RandomForest for {day_name} (not enough data for LSTM)")

                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                    model.fit(X_train_scaled, y_train)

                    # Predict and evaluate

                    y_pred = model.predict(X_test_scaled)

                    mse = mean_squared_error(y_test, y_pred)

                    mae = mean_absolute_error(y_test, y_pred)

                    dir_acc = directional_accuracy_numpy(y_test, y_pred)

                    day_model = {

                        'model': model,

                        'scaler': scaler,

                        'selected_features': selected_features,

                        'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,

                        'model_type': 'random_forest'

                    }

            # Store model and performance

            day_models[day_name] = day_model

            day_performance[day_name] = {

                'mse': mse,

                'mae': mae,

                'directional_accuracy': dir_acc,

                'count': len(test_df)

            }

            logger.info(f"{day_name} model performance - MSE: {mse:.6f}, MAE: {mae:.6f}, Dir Acc: {dir_acc:.4f}")

            # Add to DataFrame for visualization

            day_perf_df = day_perf_df.append({

                'day': day_name,

                'count': len(test_df),

                'mse': mse,

                'mae': mae,

                'directional_accuracy': dir_acc

            }, ignore_index=True)

        # Save day performance CSV

        day_perf_df.to_csv(os.path.join(RESULTS_DIR, 'day_performance.csv'), index=False)

        # Create day performance visualization

        plt.figure(figsize=(12, 6))

        ax = plt.subplot(111)

        bars = ax.bar(day_perf_df['day'], day_perf_df['directional_accuracy'], alpha=0.7)

        # Add counts and values

        for i, bar in enumerate(bars):
            height = bar.get_height()

            count = day_perf_df.iloc[i]['count']

            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,

                    f"{height:.2f}\n(n={count})", ha='center', va='bottom')

        plt.axhline(y=0.5, color='r', linestyle='--')

        plt.ylim(0, 1)

        plt.title('Directional Accuracy by Day of Week')

        plt.ylabel('Directional Accuracy')

        plt.savefig(os.path.join(RESULTS_DIR, 'day_of_week_performance.png'))

        return day_models, day_performance

    def calculate_confidence_scores(predictions, ensemble_predictions=None, model=None, X=None):

        """

        Calculate confidence scores for predictions

        Higher score = more confident prediction

        """

        # Initialize confidence scores

        confidence_scores = np.zeros_like(predictions)

        if ensemble_predictions is not None:

            # For ensemble: measure agreement between models

            # Calculate standard deviation across ensemble predictions (lower = more agreement)

            ensemble_std = np.std(ensemble_predictions, axis=0)

            # Normalize to 0-1 range (1 = highest confidence)

            max_std = np.percentile(ensemble_std, 95)  # Use 95th percentile to avoid outliers

            confidence_from_std = 1 - np.minimum(ensemble_std / max_std, 1)

            # Add weight from prediction magnitude

            abs_preds = np.abs(predictions)

            max_pred = np.percentile(abs_preds, 95)

            confidence_from_magnitude = np.minimum(abs_preds / max_pred, 1)

            # Combine factors (give more weight to ensemble agreement)

            confidence_scores = 0.7 * confidence_from_std + 0.3 * confidence_from_magnitude


        else:

            # For single model: use prediction magnitude as confidence

            abs_preds = np.abs(predictions)

            max_pred = np.percentile(abs_preds, 95)

            confidence_scores = np.minimum(abs_preds / max_pred, 1)

            # If we have a RandomForest model, add tree variance information

            if model is not None and hasattr(model, 'estimators_') and X is not None:
                # Get predictions from each tree in forest

                tree_preds = np.array([tree.predict(X) for tree in model.estimators_])

                # Calculate standard deviation across trees

                tree_std = np.std(tree_preds, axis=0)

                # Normalize to 0-1 range (1 = high confidence)

                max_tree_std = np.percentile(tree_std, 95)

                confidence_from_trees = 1 - np.minimum(tree_std / max_tree_std, 1)

                # Combine with magnitude confidence

                confidence_scores = 0.6 * confidence_scores + 0.4 * confidence_from_trees

        return confidence_scores

    def create_trading_strategy(test_df, predictions, confidence_scores=None):
        """
        Create a trading strategy with risk management
        """
        logger.info("Creating trading strategy with confidence-based position sizing")

        # Create strategy DataFrame
        strategy_df = pd.DataFrame({
            'date': test_df['arizona_time'].values[:len(predictions)],
            'actual': test_df['next_close_change_pct'].values[:len(predictions)],
            'predicted': predictions.flatten(),
            'day_of_week': test_df['day_of_week'].values[:len(predictions)]
        })

        # Map day of week to names
        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                       6: 'Sunday'}
        strategy_df['day_name'] = strategy_df['day_of_week'].map(day_mapping)

        if confidence_scores is not None:
            strategy_df['confidence'] = confidence_scores
        else:
            # Use prediction magnitude as confidence
            strategy_df['confidence'] = np.abs(strategy_df['predicted'])

        # Add regime information if available
        if 'regime' in test_df.columns:
            regime_data = test_df['regime'].values[:len(predictions)]
            strategy_df['regime'] = regime_data

        # Apply confidence threshold
        strategy_df['take_trade'] = strategy_df['confidence'] >= CONFIDENCE_THRESHOLD

        # Position sizing based on Kelly criterion
        kelly_fraction = 0.3  # Conservative Kelly
        strategy_df['position_size'] = np.where(
            strategy_df['take_trade'],
            np.abs(strategy_df['predicted']) * kelly_fraction,
            0
        )

        # Cap position size
        MAX_POSITION = 0.2  # Maximum 20% of capital
        strategy_df['position_size'] = np.minimum(strategy_df['position_size'], MAX_POSITION)

        # Determine position direction
        strategy_df['position'] = np.sign(strategy_df['predicted']) * strategy_df['position_size']

        # Calculate returns
        strategy_df['strategy_return'] = strategy_df['position'] * strategy_df['actual']
        strategy_df['cum_return'] = strategy_df['strategy_return'].cumsum()

        # Buy and hold returns
        strategy_df['buy_hold_return'] = strategy_df['actual']
        strategy_df['buy_hold_cum'] = strategy_df['buy_hold_return'].cumsum()

        # Performance metrics
        total_trades = strategy_df['take_trade'].sum()
        winning_trades = ((strategy_df['strategy_return'] > 0) & strategy_df['take_trade']).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = strategy_df.loc[
            (strategy_df['strategy_return'] > 0) & strategy_df['take_trade'], 'strategy_return'].mean() if any(
            (strategy_df['strategy_return'] > 0) & strategy_df['take_trade']) else 0
        avg_loss = strategy_df.loc[
            (strategy_df['strategy_return'] < 0) & strategy_df['take_trade'], 'strategy_return'].mean() if any(
            (strategy_df['strategy_return'] < 0) & strategy_df['take_trade']) else 0

        logger.info(f"Strategy performance - Total trades: {total_trades}, Win rate: {win_rate:.2%}")
        logger.info(f"Avg win: {avg_win:.4f}%, Avg loss: {avg_loss:.4f}%")

        # Calculate day of week performance
        day_performance = {}
        for day in range(7):
            day_mask = strategy_df['day_of_week'] == day
            if day_mask.sum() > 0:
                day_df = strategy_df[day_mask]
                day_return = day_df['strategy_return'].sum()
                day_trades = (day_df['take_trade']).sum()
                day_wins = ((day_df['strategy_return'] > 0) & day_df['take_trade']).sum()
                day_win_rate = day_wins / day_trades if day_trades > 0 else 0
                day_performance[day_mapping[day]] = {
                    'return': day_return,
                    'trades': day_trades,
                    'win_rate': day_win_rate
                }

        # Calculate regime performance if available
        regime_performance = {}
        if 'regime' in strategy_df.columns:
            for regime in strategy_df['regime'].unique():
                regime_mask = strategy_df['regime'] == regime
                if regime_mask.sum() > 0:
                    regime_df = strategy_df[regime_mask]
                    regime_return = regime_df['strategy_return'].sum()
                    regime_trades = (regime_df['take_trade']).sum()
                    regime_wins = ((regime_df['strategy_return'] > 0) & regime_df['take_trade']).sum()
                    regime_win_rate = regime_wins / regime_trades if regime_trades > 0 else 0
                    regime_names = {0: 'Normal', 1: 'Trending', 2: 'Mean-Rev', 3: 'Volatile'}
                    regime_name = regime_names.get(regime, f"Regime {regime}")
                    regime_performance[regime_name] = {
                        'return': regime_return,
                        'trades': regime_trades,
                        'win_rate': regime_win_rate
                    }

        # Save strategy results
        strategy_results = {
            'trades': {
                'total': int(total_trades),
                'wins': int(winning_trades),
                'losses': int(total_trades - winning_trades),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss)
            },
            'day_performance': day_performance,
            'regime_performance': regime_performance,
            'strategy_df': strategy_df
        }

        # Plot strategy performance
        plt.figure(figsize=(12, 8))

        # Plot cumulative returns
        plt.subplot(2, 1, 1)
        plt.plot(strategy_df['date'], strategy_df['cum_return'], label='Strategy', color='blue')
        plt.plot(strategy_df['date'], strategy_df['buy_hold_cum'], label='Buy & Hold', color='green', alpha=0.6)
        plt.title('Strategy Performance')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot confidence and trades
        plt.subplot(2, 1, 2)
        plt.scatter(strategy_df['date'], strategy_df['confidence'], alpha=0.5, label='Confidence', color='gray')

        # Highlight trades
        trades = strategy_df[strategy_df['take_trade']]
        won_trades = trades[trades['strategy_return'] > 0]
        lost_trades = trades[trades['strategy_return'] < 0]
        plt.scatter(won_trades['date'], won_trades['confidence'], color='green', label='Winning Trade')
        plt.scatter(lost_trades['date'], lost_trades['confidence'], color='red', label='Losing Trade')
        plt.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'confidence_analysis.png'))

        # Save day performance visualization
        day_perf = pd.DataFrame([
            {'day': day, 'return': data['return'], 'win_rate': data['win_rate'], 'trades': data['trades']}
            for day, data in day_performance.items()
        ])

        if not day_perf.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(day_perf['day'], day_perf['return'], alpha=0.7)
            plt.title('Strategy Return by Day of Week')
            plt.ylabel('Cumulative Return (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(RESULTS_DIR, 'strategy_day_performance.png'))

        # Fix: Add return statement that was missing
        return strategy_results


def plot_feature_importance(importance_values, feature_names, top_n=30):
    """Plot feature importance"""

    # Create DataFrame for easier manipulation

    importance_df = pd.DataFrame({

        'Feature': feature_names,

        'Importance': importance_values

    })

    # Sort by importance

    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Take top N features

    top_features = importance_df.head(top_n)

    # Plot

    plt.figure(figsize=(12, 8))

    bars = plt.barh(top_features['Feature'], top_features['Importance'])

    # Add values

    for bar in bars:
        width = bar.get_width()

        plt.text(width + 0.0005, bar.get_y() + bar.get_height() / 2,

                 f"{width:.4f}", ha='left', va='center')

    plt.title(f'Top {top_n} Feature Importance')

    plt.xlabel('Importance')

    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))

    # Save to CSV

    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)

    return top_features


def train_final_model(model, X_train, y_train, X_val, y_val, is_sequence=False, batch_size=32, epochs=100):
    """

    Train the final model with all callbacks

    """

    if not is_sequence and not isinstance(model, (Sequential, Model)):
        # Traditional ML model

        logger.info("Training final traditional model")

        model.fit(X_train, y_train)

        # Evaluate on validation set

        val_pred = model.predict(X_val)

        val_mse = mean_squared_error(y_val, val_pred)

        val_mae = mean_absolute_error(y_val, val_pred)

        val_dir_acc = directional_accuracy_numpy(y_val, val_pred)

        logger.info(f"Validation - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, Dir Acc: {val_dir_acc:.4f}")

        return model, {'val_mse': val_mse, 'val_mae': val_mae, 'val_dir_acc': val_dir_acc}

    # Neural network model

    logger.info("Training final neural network model")

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

        filepath=os.path.join(MODEL_DIR, 'best_model.h5'),

        monitor='val_loss',

        save_best_only=True,

        verbose=1

    )

    tensorboard = TensorBoard(

        log_dir=os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),

        histogram_freq=1

    )

    # Train model

    history = model.fit(

        X_train, y_train,

        validation_data=(X_val, y_val),

        epochs=epochs,

        batch_size=batch_size,

        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],

        verbose=1

    )

    return model, history


def evaluate_model(model, X_test, y_test, is_sequence=False, is_ensemble=False, ensemble_models=None,
                   ensemble_weights=None):
    """
    Evaluate the model on test data
    """
    # Make predictions
    if is_ensemble and ensemble_models is not None:
        # Ensemble predictions
        ensemble_predictions = []
        for m in ensemble_models:
            if is_sequence:
                pred = m.predict(X_test)
            else:
                pred = m.predict(X_test).flatten() if hasattr(m, 'predict') else m.predict(X_test)
            ensemble_predictions.append(pred.flatten())

        # Calculate weighted predictions
        ensemble_predictions = np.array(ensemble_predictions)
        y_pred = np.zeros(ensemble_predictions.shape[1])
        for i, weight in enumerate(ensemble_weights):
            y_pred += weight * ensemble_predictions[i]

    elif hasattr(model, 'predict'):
        # Neural network or sklearn model
        y_pred = model.predict(X_test)
        if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
    else:
        # Unknown model type
        logger.error("Unknown model type for prediction")
        return None

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    try:
        r2 = r2_score(y_test, y_pred)
    except:
        r2 = 0  # Default if r2_score fails

    # Directional accuracy
    dir_acc = np.mean((np.sign(y_test) == np.sign(y_pred)).astype(int))

    # Log metrics
    logger.info(f"Test Metrics:")
    logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}, R²: {r2:.6f}")
    logger.info(f"Directional Accuracy: {dir_acc:.4f}")

    # Evaluate different prediction magnitudes
    # Create bins based on predicted magnitude
    abs_pred = np.abs(y_pred)

    # Define bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, float('inf')]
    bin_labels = ['0-0.1%', '0.1-0.2%', '0.2-0.3%', '0.3-0.4%', '0.4-0.5%', '>0.5%']

    # Bin the predictions
    bin_indices = np.digitize(abs_pred, bins[1:])

    # Calculate metrics by bin
    bin_metrics = []
    for i, label in enumerate(bin_labels):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_y_test = y_test[mask]
            bin_y_pred = y_pred[mask]
            bin_dir_acc = np.mean((np.sign(bin_y_test) == np.sign(bin_y_pred)).astype(int))
            bin_count = np.sum(mask)
            bin_metrics.append({
                'bin': label,
                'count': bin_count,
                'dir_acc': bin_dir_acc
            })
            logger.info(f"Bin {label}: {bin_count} samples, Dir Acc: {bin_dir_acc:.4f}")

    # Ensure the Results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot accuracy by magnitude
    plt.figure(figsize=(10, 6))
    bins = [m['bin'] for m in bin_metrics]
    accs = [m['dir_acc'] for m in bin_metrics]
    counts = [m['count'] for m in bin_metrics]
    bars = plt.bar(bins, accs)

    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"n={counts[i]}", ha='center', va='bottom')

    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Directional Accuracy by Prediction Magnitude')
    plt.ylabel('Directional Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Make sure the directory exists before saving
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_by_magnitude.png'))

    # Return metrics and predictions
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': dir_acc,
        'predictions': y_pred,
        'actual': y_test,
        'bin_metrics': bin_metrics
    }

def train_day_specific_models(df_5pm):
    """
    Train separate models for each day of the week to address the Wednesday performance issue
    """
    logger.info("Training day-specific models")
    day_models = {}
    day_performance = {}
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Create a DataFrame to store day performance for visualization
    day_perf_df = pd.DataFrame(columns=['day', 'count', 'mse', 'mae', 'directional_accuracy'])

    for day_idx, day_name in day_mapping.items():
        day_df = df_5pm[df_5pm['day_of_week'] == day_idx].copy()
        if len(day_df) < 30:
            logger.warning(f"Not enough data for {day_name}, skipping")
            continue

        logger.info(f"Training model for {day_name} with {len(day_df)} samples")

        # Split, prepare, and train model for this day
        train_df, val_df, test_df = time_series_split(day_df)
        X_train, y_train, feature_names = prepare_features_and_targets(train_df)
        X_val, y_val, _ = prepare_features_and_targets(val_df)
        X_test, y_test, _ = prepare_features_and_targets(test_df)

        # Use more features for Wednesday (which has poor performance)
        n_features = 80 if day_name == 'Wednesday' else 50

        # Select features specifically for this day
        if USE_FEATURE_SELECTION:
            X_train_selected, selected_indices, selected_features = select_features(
                X_train, y_train, feature_names, method=FEATURE_SELECTION_METHOD, n_features=n_features
            )
            X_val_selected = X_val[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
        else:
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test
            selected_features = feature_names

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train_selected, X_val_selected, X_test_selected, scaler_type='robust'
        )

        # For Wednesday, use a more robust model or ensemble
        if day_name == 'Wednesday':
            logger.info("Using RandomForest for Wednesday to improve performance")
            model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                         min_samples_leaf=10, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            dir_acc = directional_accuracy_numpy(y_test, y_pred)

            day_model = {
                'model': model,
                'scaler': scaler,
                'selected_features': selected_features,
                'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,
                'model_type': 'random_forest'
            }
        else:
            # For other days, use LSTM if enough data, otherwise RF
            if len(X_train) >= 100:
                # Create sequences for LSTM
                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
                X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val)
                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)

                logger.info(f"Training LSTM for {day_name}")
                model = build_lstm_model(
                    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                    complexity='medium',
                    dropout_rate=0.4  # Higher dropout for regularization
                )

                model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=100,
                    batch_size=32,
                    callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                    verbose=1
                )

                # Predict and evaluate
                y_pred = model.predict(X_test_seq).flatten()
                mse = mean_squared_error(y_test_seq, y_pred)
                mae = mean_absolute_error(y_test_seq, y_pred)
                dir_acc = directional_accuracy_numpy(y_test_seq, y_pred)

                day_model = {
                    'model': model,
                    'scaler': scaler,
                    'selected_features': selected_features,
                    'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,
                    'model_type': 'lstm',
                    'is_sequence': True
                }
            else:
                logger.info(f"Using RandomForest for {day_name} (not enough data for LSTM)")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                dir_acc = directional_accuracy_numpy(y_test, y_pred)

                day_model = {
                    'model': model,
                    'scaler': scaler,
                    'selected_features': selected_features,
                    'selected_indices': selected_indices if USE_FEATURE_SELECTION else None,
                    'model_type': 'random_forest'
                }

        # Store model and performance
        day_models[day_name] = day_model
        day_performance[day_name] = {
            'mse': mse,
            'mae': mae,
            'directional_accuracy': dir_acc,
            'count': len(test_df)
        }

        logger.info(f"{day_name} model performance - MSE: {mse:.6f}, MAE: {mae:.6f}, Dir Acc: {dir_acc:.4f}")

        # Add to DataFrame for visualization
        # Using pandas concat instead of deprecated append
        new_row = pd.DataFrame({
            'day': [day_name],
            'count': [len(test_df)],
            'mse': [mse],
            'mae': [mae],
            'directional_accuracy': [dir_acc]
        })
        day_perf_df = pd.concat([day_perf_df, new_row], ignore_index=True)

    # Save day performance CSV
    day_perf_df.to_csv(os.path.join(RESULTS_DIR, 'day_performance.csv'), index=False)

    # Create day performance visualization
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    bars = ax.bar(day_perf_df['day'], day_perf_df['directional_accuracy'], alpha=0.7)

    # Add counts and values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = day_perf_df.iloc[i]['count']
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{height:.2f}\n(n={count})", ha='center', va='bottom')

    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.ylim(0, 1)
    plt.title('Directional Accuracy by Day of Week')
    plt.ylabel('Directional Accuracy')
    plt.savefig(os.path.join(RESULTS_DIR, 'day_of_week_performance.png'))

    return day_models, day_performance

def save_model_and_metadata(model, scaler, feature_list, selected_indices, best_params, test_metrics,

                            is_ensemble=False, ensemble_models=None, ensemble_weights=None,

                            day_models=None):
    """

    Save model, scaler, feature list, and metadata

    """

    # Save model architecture as JSON for better portability

    if is_ensemble and ensemble_models is not None:

        # Create ensemble directory

        ensemble_dir = os.path.join(MODEL_DIR, 'ensemble')

        os.makedirs(ensemble_dir, exist_ok=True)

        # Save each model in the ensemble

        for i, model in enumerate(ensemble_models):

            if hasattr(model, 'save'):

                model.save(os.path.join(ensemble_dir, f'model_{i}.h5'))

            else:

                # Pickle sklearn model

                with open(os.path.join(ensemble_dir, f'model_{i}.pkl'), 'wb') as f:

                    pickle.dump(model, f)

        # Save ensemble weights

        with open(os.path.join(ensemble_dir, 'ensemble_weights.json'), 'w') as f:

            json.dump(ensemble_weights, f)

    else:

        # Save single model

        if hasattr(model, 'save'):

            model.save(os.path.join(MODEL_DIR, 'final_model.h5'))

            # Save model architecture separately

            if hasattr(model, 'to_json'):
                model_json = model.to_json()

                with open(os.path.join(MODEL_DIR, 'model_architecture.json'), 'w') as f:
                    f.write(model_json)

        else:

            # Pickle sklearn model

            with open(os.path.join(MODEL_DIR, 'final_model.pkl'), 'wb') as f:

                pickle.dump(model, f)

    # Save scaler

    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:

        pickle.dump(scaler, f)

    # Save feature list and selected indices

    feature_data = {

        'feature_list': feature_list,

        'selected_indices': selected_indices

    }

    with open(os.path.join(MODEL_DIR, 'feature_data.pkl'), 'wb') as f:

        pickle.dump(feature_data, f)

    # Save day-specific models if available

    if day_models is not None:

        for day, model_data in day_models.items():

            day_dir = os.path.join(DAY_MODEL_DIR, day)

            os.makedirs(day_dir, exist_ok=True)

            # Save model

            model = model_data['model']

            if hasattr(model, 'save'):

                model.save(os.path.join(day_dir, 'model.h5'))

            else:

                with open(os.path.join(day_dir, 'model.pkl'), 'wb') as f:

                    pickle.dump(model, f)

            # Save scaler

            with open(os.path.join(day_dir, 'scaler.pkl'), 'wb') as f:

                pickle.dump(model_data['scaler'], f)

            # Save feature data

            day_feature_data = {

                'selected_features': model_data['selected_features'],

                'selected_indices': model_data.get('selected_indices', None),

                'model_type': model_data.get('model_type', 'unknown')

            }

            with open(os.path.join(day_dir, 'feature_data.pkl'), 'wb') as f:

                pickle.dump(day_feature_data, f)

    # Convert test_metrics to JSON-serializable format

    serializable_metrics = {}

    for k, v in test_metrics.items():

        if k not in ['predictions', 'actual', 'bin_metrics']:

            if isinstance(v, (np.float32, np.float64)):

                serializable_metrics[k] = float(v)

            elif isinstance(v, (np.int32, np.int64)):

                serializable_metrics[k] = int(v)

            elif isinstance(v, np.bool_):

                serializable_metrics[k] = bool(v)

            elif isinstance(v, np.ndarray):

                continue  # Skip arrays

            else:

                serializable_metrics[k] = v

    # Make sure hyperparameters are also serializable

    serializable_params = {}

    for k, v in best_params.items():

        if isinstance(v, (np.bool_, bool)):

            serializable_params[k] = bool(v)

        elif isinstance(v, (np.int32, np.int64)):

            serializable_params[k] = int(v)

        elif isinstance(v, (np.float32, np.float64)):

            serializable_params[k] = float(v)

        else:

            serializable_params[k] = v

    # Create metadata dictionary with JSON-safe values

    metadata = {

        'hyperparameters': serializable_params,

        'test_metrics': serializable_metrics,

        'feature_count': int(len(feature_list)),

        'selected_feature_count': int(len(selected_indices)) if selected_indices is not None else int(
            len(feature_list)),

        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        'is_ensemble': bool(is_ensemble),

        'ensemble_size': int(len(ensemble_models) if is_ensemble and ensemble_models else 1),

        'day_specific_models': bool(day_models is not None),

        'model_type': serializable_params.get('model_type', 'ensemble' if is_ensemble else 'unknown'),

        'symbol': SYMBOL,

        'lookback': int(LOOKBACK),

        'handle_outliers': bool(HANDLE_OUTLIERS),

        'use_feature_selection': bool(USE_FEATURE_SELECTION),

        'feature_selection_method': FEATURE_SELECTION_METHOD if USE_FEATURE_SELECTION else None,

        'use_day_specific_models': bool(USE_DAY_SPECIFIC_MODELS),

        'weighted_loss': bool(WEIGHTED_LOSS)

    }

    # Save metadata

    try:

        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
            json.dump(make_json_serializable(metadata), f, indent=4)

        logger.info(f"Model and metadata saved to {MODEL_DIR}")

    except TypeError as e:

        logger.error(f"Error saving metadata: {e}")

        # Print the metadata content for debugging

        for key, value in metadata.items():

            logger.info(f"Metadata key '{key}' has type {type(value)}")

            if isinstance(value, dict):

                for k, v in value.items():
                    logger.info(f"  Subkey '{k}' has type {type(v)}")

    # Save validation summary

    validation_summary = {

        'walk_forward_metrics': serializable_metrics,

        'top_features': feature_list[:5] if selected_indices is None else [feature_list[i] for i in
                                                                           selected_indices[:5]],

        'best_day': 'Tuesday',  # Based on your data

        'best_month': 'Nov',

        'best_regime': 'Normal'

    }

    with open(os.path.join(RESULTS_DIR, 'validation_summary.json'), 'w') as f:
        json.dump(make_json_serializable(validation_summary), f, indent=4)

def optimize_with_optuna(X_train, y_train, X_val, y_val, model_type, is_sequence=True, n_trials=50):
    """Run hyperparameter optimization with Optuna"""
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, falling back to grid search")
        return hyperparameter_grid_search(X_train, y_train, X_val, y_val, model_type, is_sequence)

    logger.info(f"Starting Optuna hyperparameter optimization for {model_type} with {n_trials} trials")

    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    # Create objective function
    objective = lambda trial: optuna_objective(
        trial, X_train, y_train, X_val, y_val, model_type, is_sequence
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Best parameters for {model_type}: {best_params}")
    logger.info(f"Best directional accuracy: {best_value:.4f}")

    # Build model with best parameters
    if model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and is_sequence:
        input_shape = (X_train.shape[1], X_train.shape[2])
        best_model = build_model_by_type(model_type, input_shape, best_params)
    elif model_type == 'simple_nn' or (model_type in ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention'] and not is_sequence):
        input_shape = X_train.shape[1]
        best_model = build_simple_nn_model(
            input_shape,
            dropout_rate=best_params.get('dropout_rate', 0.3),
            learning_rate=best_params.get('learning_rate', 0.001)
        )
    else:
        best_model = build_model_by_type(model_type, params=best_params)

    return best_model, best_params

def build_stacked_ensemble(X_train, y_train, X_val, y_val, base_models=None):
    """
    Build a stacked ensemble with diverse base models
    Uses base models' predictions as features for a meta-learner
    """
    logger.info("Building stacked ensemble model")

    if base_models is None:
        # Create diverse base models
        base_models = []
        # Add traditional models
        base_models.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42)))
        base_models.append(('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42)))
        base_models.append(('et', ExtraTreesRegressor(n_estimators=100, random_state=42)))

        # Add neural network if we have enough data
        if len(X_train) >= 500:
            nn_model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            nn_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
            base_models.append(('nn', nn_model))

    # Train base models and get their predictions
    base_predictions_train = np.zeros((X_train.shape[0], len(base_models)))
    base_predictions_val = np.zeros((X_val.shape[0], len(base_models)))
    trained_models = []

    for i, (name, model) in enumerate(base_models):
        logger.info(f"Training base model: {name}")
        if isinstance(model, (Sequential, Model)):
            # Neural network model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            base_predictions_train[:, i] = model.predict(X_train).flatten()
            base_predictions_val[:, i] = model.predict(X_val).flatten()
        else:
            # Sklearn model
            model.fit(X_train, y_train)
            base_predictions_train[:, i] = model.predict(X_train)
            base_predictions_val[:, i] = model.predict(X_val)

        trained_models.append((name, model))

        # Evaluate individual model
        if isinstance(model, (Sequential, Model)):
            val_metrics = model.evaluate(X_val, y_val, verbose=0)
            val_loss = val_metrics[0]
            val_dir_acc = val_metrics[2]  # directional_accuracy is 3rd metric
        else:
            val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            val_dir_acc = directional_accuracy_numpy(y_val, val_pred)

        logger.info(f"  {name} - Val Loss: {val_loss:.6f}, Dir Acc: {val_dir_acc:.4f}")

    # Add original features to meta-features
    meta_features_train = np.hstack([base_predictions_train, X_train[:, :20]])  # Use top 20 original features
    meta_features_val = np.hstack([base_predictions_val, X_val[:, :20]])

    # Train meta-learner
    logger.info("Training meta-learner")
    meta_learner = GradientBoostingRegressor(n_estimators=100, random_state=42)
    meta_learner.fit(meta_features_train, y_train)

    # Evaluate ensemble
    meta_pred_val = meta_learner.predict(meta_features_val)
    ensemble_mse = mean_squared_error(y_val, meta_pred_val)
    ensemble_dir_acc = directional_accuracy_numpy(y_val, meta_pred_val)

    logger.info(f"Stacked Ensemble - Val MSE: {ensemble_mse:.6f}, Dir Acc: {ensemble_dir_acc:.4f}")

    # Store models and metadata
    ensemble = {
        'base_models': trained_models,
        'meta_learner': meta_learner,
        'performance': {
            'mse': ensemble_mse,
            'directional_accuracy': ensemble_dir_acc
        }
    }

    return ensemble

def predict_with_stacked_ensemble(ensemble, X):
    """Make predictions with the stacked ensemble"""
    base_models = ensemble['base_models']
    meta_learner = ensemble['meta_learner']

    # Get base model predictions
    base_predictions = np.zeros((X.shape[0], len(base_models)))
    for i, (name, model) in enumerate(base_models):
        if isinstance(model, (Sequential, Model)):
            base_predictions[:, i] = model.predict(X).flatten()
        else:
            base_predictions[:, i] = model.predict(X)

    # Combine with top 20 original features for meta-learner
    meta_features = np.hstack([base_predictions, X[:, :20]])

    # Make final prediction
    return meta_learner.predict(meta_features)


def calculate_confidence_scores(predictions, ensemble_predictions=None, model=None, X=None):
    """
    Calculate confidence scores for predictions
    Higher score = more confident prediction
    """
    # Initialize confidence scores
    confidence_scores = np.zeros_like(predictions)

    if ensemble_predictions is not None:
        # For ensemble: measure agreement between models
        # Calculate standard deviation across ensemble predictions (lower = more agreement)
        ensemble_std = np.std(ensemble_predictions, axis=0)

        # Normalize to 0-1 range (1 = highest confidence)
        max_std = np.percentile(ensemble_std, 95)  # Use 95th percentile to avoid outliers
        confidence_from_std = 1 - np.minimum(ensemble_std / max_std, 1)

        # Add weight from prediction magnitude
        abs_preds = np.abs(predictions)
        max_pred = np.percentile(abs_preds, 95)
        confidence_from_magnitude = np.minimum(abs_preds / max_pred, 1)

        # Combine factors (give more weight to ensemble agreement)
        confidence_scores = 0.7 * confidence_from_std + 0.3 * confidence_from_magnitude

    else:
        # For single model: use prediction magnitude as confidence
        abs_preds = np.abs(predictions)
        max_pred = np.percentile(abs_preds, 95)
        confidence_scores = np.minimum(abs_preds / max_pred, 1)

        # If we have a RandomForest model, add tree variance information
        if model is not None and hasattr(model, 'estimators_') and X is not None:
            # Get predictions from each tree in forest
            tree_preds = np.array([tree.predict(X) for tree in model.estimators_])

            # Calculate standard deviation across trees
            tree_std = np.std(tree_preds, axis=0)

            # Normalize to 0-1 range (1 = high confidence)
            max_tree_std = np.percentile(tree_std, 95)
            confidence_from_trees = 1 - np.minimum(tree_std / max_tree_std, 1)

            # Combine with magnitude confidence
            confidence_scores = 0.6 * confidence_scores + 0.4 * confidence_from_trees

    return confidence_scores


def create_trading_strategy(test_df, predictions, confidence_scores=None):
    """
    Create a trading strategy with risk management
    """
    logger.info("Creating trading strategy with confidence-based position sizing")

    # Create strategy DataFrame
    strategy_df = pd.DataFrame({
        'date': test_df['arizona_time'].values[:len(predictions)],
        'actual': test_df['next_close_change_pct'].values[:len(predictions)],
        'predicted': predictions.flatten(),
        'day_of_week': test_df['day_of_week'].values[:len(predictions)]
    })

    # Map day of week to names
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    strategy_df['day_name'] = strategy_df['day_of_week'].map(day_mapping)

    if confidence_scores is not None:
        strategy_df['confidence'] = confidence_scores
    else:
        # Use prediction magnitude as confidence
        strategy_df['confidence'] = np.abs(strategy_df['predicted'])

    # Add regime information if available
    if 'regime' in test_df.columns:
        regime_data = test_df['regime'].values[:len(predictions)]
        strategy_df['regime'] = regime_data

    # Apply confidence threshold
    strategy_df['take_trade'] = strategy_df['confidence'] >= CONFIDENCE_THRESHOLD

    # Position sizing based on Kelly criterion
    kelly_fraction = 0.3  # Conservative Kelly
    strategy_df['position_size'] = np.where(
        strategy_df['take_trade'],
        np.abs(strategy_df['predicted']) * kelly_fraction,
        0
    )

    # Cap position size
    MAX_POSITION = 0.2  # Maximum 20% of capital
    strategy_df['position_size'] = np.minimum(strategy_df['position_size'], MAX_POSITION)

    # Determine position direction
    strategy_df['position'] = np.sign(strategy_df['predicted']) * strategy_df['position_size']

    # Calculate returns
    strategy_df['strategy_return'] = strategy_df['position'] * strategy_df['actual']
    strategy_df['cum_return'] = strategy_df['strategy_return'].cumsum()

    # Buy and hold returns
    strategy_df['buy_hold_return'] = strategy_df['actual']
    strategy_df['buy_hold_cum'] = strategy_df['buy_hold_return'].cumsum()

    # Performance metrics
    total_trades = strategy_df['take_trade'].sum()
    winning_trades = ((strategy_df['strategy_return'] > 0) & strategy_df['take_trade']).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = strategy_df.loc[
        (strategy_df['strategy_return'] > 0) & strategy_df['take_trade'], 'strategy_return'].mean() \
        if any((strategy_df['strategy_return'] > 0) & strategy_df['take_trade']) else 0

    avg_loss = strategy_df.loc[
        (strategy_df['strategy_return'] < 0) & strategy_df['take_trade'], 'strategy_return'].mean() \
        if any((strategy_df['strategy_return'] < 0) & strategy_df['take_trade']) else 0

    logger.info(f"Strategy performance - Total trades: {total_trades}, Win rate: {win_rate:.2%}")
    logger.info(f"Avg win: {avg_win:.4f}%, Avg loss: {avg_loss:.4f}%")

    # Calculate day of week performance
    day_performance = {}
    for day in range(7):
        day_mask = strategy_df['day_of_week'] == day
        if day_mask.sum() > 0:
            day_df = strategy_df[day_mask]
            day_return = day_df['strategy_return'].sum()
            day_trades = (day_df['take_trade']).sum()
            day_wins = ((day_df['strategy_return'] > 0) & day_df['take_trade']).sum()
            day_win_rate = day_wins / day_trades if day_trades > 0 else 0
            day_performance[day_mapping[day]] = {
                'return': day_return,
                'trades': day_trades,
                'win_rate': day_win_rate
            }

    # Calculate regime performance if available
    regime_performance = {}
    if 'regime' in strategy_df.columns:
        for regime in strategy_df['regime'].unique():
            regime_mask = strategy_df['regime'] == regime
            if regime_mask.sum() > 0:
                regime_df = strategy_df[regime_mask]
                regime_return = regime_df['strategy_return'].sum()
                regime_trades = (regime_df['take_trade']).sum()
                regime_wins = ((regime_df['strategy_return'] > 0) & regime_df['take_trade']).sum()
                regime_win_rate = regime_wins / regime_trades if regime_trades > 0 else 0
                regime_names = {0: 'Normal', 1: 'Trending', 2: 'Mean-Rev', 3: 'Volatile'}
                regime_name = regime_names.get(regime, f"Regime {regime}")
                regime_performance[regime_name] = {
                    'return': regime_return,
                    'trades': regime_trades,
                    'win_rate': regime_win_rate
                }

    # Save strategy results
    strategy_results = {
        'trades': {
            'total': int(total_trades),
            'wins': int(winning_trades),
            'losses': int(total_trades - winning_trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss)
        },
        'day_performance': day_performance,
        'regime_performance': regime_performance,
        'strategy_df': strategy_df
    }

    # Plot strategy performance
    plt.figure(figsize=(12, 8))

    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(strategy_df['date'], strategy_df['cum_return'], label='Strategy', color='blue')
    plt.plot(strategy_df['date'], strategy_df['buy_hold_cum'], label='Buy & Hold', color='green', alpha=0.6)
    plt.title('Strategy Performance')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot confidence and trades
    plt.subplot(2, 1, 2)
    plt.scatter(strategy_df['date'], strategy_df['confidence'], alpha=0.5, label='Confidence', color='gray')

    # Highlight trades
    trades = strategy_df[strategy_df['take_trade']]
    won_trades = trades[trades['strategy_return'] > 0]
    lost_trades = trades[trades['strategy_return'] < 0]
    plt.scatter(won_trades['date'], won_trades['confidence'], color='green', label='Winning Trade')
    plt.scatter(lost_trades['date'], lost_trades['confidence'], color='red', label='Losing Trade')
    plt.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
    plt.ylabel('Confidence Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confidence_analysis.png'))

    # Save day performance visualization
    day_perf = pd.DataFrame([
        {'day': day, 'return': data['return'], 'win_rate': data['win_rate'], 'trades': data['trades']}
        for day, data in day_performance.items()
    ])

    if not day_perf.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(day_perf['day'], day_perf['return'], alpha=0.7)
        plt.title('Strategy Return by Day of Week')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, 'strategy_day_performance.png'))

    return strategy_results

def main():
    """Main training function with improved pipeline"""

    # Ensure required directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DAY_MODEL_DIR, exist_ok=True)

    logger.info("Ensuring all required directories exist")

    # MT5 connection params

    account = 90933473

    password = "NhXgR*3g"

    server = "MetaQuotes-Demo"

    # Connect to MT5

    if not connect_to_mt5(account, password, server):
        return

    try:

        # Define date range for historical data (5 years - increased for more training data)

        end_date = datetime.now(ARIZONA_TZ)

        start_date = end_date - timedelta(days=5 * 365)  # 5 years of data

        # Get historical data

        logger.info(f"Fetching historical data from {start_date} to {end_date}")

        df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)

        if df is None:
            return

        # Filter for 5 PM Arizona time

        df_5pm = filter_5pm_data(df)

        # Add economic indicators if enabled

        if ADD_MACROECONOMIC:
            logger.info("Loading and adding macroeconomic indicators")

            macro_df = load_economic_indicators()

            df_5pm = add_macroeconomic_data(df_5pm, macro_df)

        # Feature engineering

        logger.info("Adding datetime features...")

        df_5pm = add_datetime_features(df_5pm)

        # Add market regime detection if enabled

        if MARKET_REGIMES:
            logger.info("Detecting market regimes...")

            df_5pm = detect_market_regime(df_5pm)

        # Add wavelet features if enabled

        if USE_WAVELET:
            logger.info("Adding wavelet decomposition features...")

            df_5pm = add_wavelet_features(df_5pm)

        logger.info("Adding technical indicators...")

        df_5pm = add_technical_indicators(df_5pm)

        logger.info("Adding lagged features...")

        df_5pm = add_lagged_features(df_5pm)

        logger.info("Adding target variables...")

        df_5pm = add_target_variables(df_5pm)

        # Handle outliers if enabled

        if HANDLE_OUTLIERS:
            logger.info("Handling outliers in target variable...")

            columns_to_handle = ['next_close_change_pct', 'change_future_2_pct', 'change_future_3_pct',

                                 'close_diff_pct', 'volatility_20']

            df_5pm = handle_outliers(df_5pm, columns_to_handle)

        # Train day-specific models if enabled

        if USE_DAY_SPECIFIC_MODELS:

            logger.info("Training day-specific models...")

            day_models, day_performance = train_day_specific_models(df_5pm)

        else:

            day_models = None

            day_performance = None

        # Split data

        train_df, val_df, test_df = time_series_split(df_5pm)

        # Prepare features and targets

        X_train, y_train, feature_list = prepare_features_and_targets(train_df)

        X_val, y_val, _ = prepare_features_and_targets(val_df)

        X_test, y_test, _ = prepare_features_and_targets(test_df)

        # Feature selection if enabled

        if USE_FEATURE_SELECTION:

            logger.info(f"Performing feature selection using {FEATURE_SELECTION_METHOD}...")

            X_train_selected, selected_indices, selected_feature_names = select_features(

                X_train, y_train, feature_list, method=FEATURE_SELECTION_METHOD, n_features=MIN_FEATURES

            )

            X_val_selected = X_val[:, selected_indices]

            X_test_selected = X_test[:, selected_indices]

        else:

            X_train_selected = X_train

            X_val_selected = X_val

            X_test_selected = X_test

            selected_indices = None

            selected_feature_names = feature_list

        # Scale features

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(

            X_train_selected, X_val_selected, X_test_selected, scaler_type='robust'

        )

        # Apply SMOTE for class balancing if enabled and available

        if USE_SMOTE and 'next_direction' in train_df.columns:

            try:

                logger.info("Applying SMOTE to balance classes...")

                smote = SMOTE(random_state=42)

                X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

                logger.info(
                    f"SMOTE applied - Original shape: {X_train_scaled.shape}, New shape: {X_train_scaled_smote.shape}")

                # Use SMOTE-enhanced data

                X_train_scaled = X_train_scaled_smote

                y_train = y_train_smote

            except Exception as e:

                logger.warning(f"SMOTE failed: {e}, continuing with original data")

        # Try simpler models first if enabled

        if SIMPLER_MODELS_FIRST:

            logger.info("Testing simpler models first...")

            # Create sequences for LSTM/GRU models later

            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, LOOKBACK)

            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, LOOKBACK)

            # Train traditional models

            model_performances = {}

            traditional_models = []

            # Random Forest baseline

            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

            rf_model.fit(X_train_scaled, y_train)

            rf_pred = rf_model.predict(X_val_scaled)

            rf_mse = mean_squared_error(y_val, rf_pred)

            rf_dir_acc = directional_accuracy_numpy(y_val, rf_pred)

            model_performances['random_forest'] = {

                'mse': rf_mse,

                'dir_acc': rf_dir_acc

            }

            traditional_models.append(('random_forest', rf_model))

            logger.info(f"Random Forest - MSE: {rf_mse:.6f}, Dir Acc: {rf_dir_acc:.4f}")

            # GBM baseline

            gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

            gbm_model.fit(X_train_scaled, y_train)

            gbm_pred = gbm_model.predict(X_val_scaled)

            gbm_mse = mean_squared_error(y_val, gbm_pred)

            gbm_dir_acc = directional_accuracy_numpy(y_val, gbm_pred)

            model_performances['gbm'] = {

                'mse': gbm_mse,

                'dir_acc': gbm_dir_acc

            }

            traditional_models.append(('gbm', gbm_model))

            logger.info(f"Gradient Boosting - MSE: {gbm_mse:.6f}, Dir Acc: {gbm_dir_acc:.4f}")

            # Simple NN baseline

            simple_nn = build_simple_nn_model(X_train_scaled.shape[1])

            simple_nn.fit(X_train_scaled, y_train,

                          validation_data=(X_val_scaled, y_val),

                          epochs=50, batch_size=32,

                          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],

                          verbose=0)

            nn_metrics = simple_nn.evaluate(X_val_scaled, y_val, verbose=0)

            nn_mse = nn_metrics[0]

            nn_dir_acc = nn_metrics[2]  # directional_accuracy is 3rd metric

            model_performances['simple_nn'] = {

                'mse': nn_mse,

                'dir_acc': nn_dir_acc

            }

            traditional_models.append(('simple_nn', simple_nn))

            logger.info(f"Simple NN - MSE: {nn_mse:.6f}, Dir Acc: {nn_dir_acc:.4f}")

            # LSTM baseline

            lstm_model = build_lstm_model(

                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),

                complexity='low'

            )

            lstm_model.fit(X_train_seq, y_train_seq,

                           validation_data=(X_val_seq, y_val_seq),

                           epochs=50, batch_size=32,

                           callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],

                           verbose=0)

            lstm_metrics = lstm_model.evaluate(X_val_seq, y_val_seq, verbose=0)

            lstm_mse = lstm_metrics[0]

            lstm_dir_acc = lstm_metrics[2]  # directional_accuracy is 3rd metric

            model_performances['lstm'] = {

                'mse': lstm_mse,

                'dir_acc': lstm_dir_acc

            }

            # Advanced: Find the best model type based on directional accuracy

            best_model_type = max(model_performances.items(), key=lambda x: x[1]['dir_acc'])[0]

            best_dir_acc = model_performances[best_model_type]['dir_acc']

            logger.info(f"Best model type: {best_model_type} with directional accuracy {best_dir_acc:.4f}")

            # Decide whether to use traditional or neural network model

            if best_model_type in ['random_forest', 'gbm']:

                logger.info("Traditional model performs better, using it as primary model type")

                model_type = best_model_type

                is_sequence = False

            else:

                # For neural networks, use LSTM as default

                logger.info("Neural network model performs better, using LSTM as primary model type")

                model_type = 'lstm'

                is_sequence = True

        else:

            # Default to LSTM

            model_type = 'lstm'

            is_sequence = True

            # Create sequences for LSTM/GRU models

            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, LOOKBACK)

            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, LOOKBACK)

        # Optimize hyperparameters

        if USE_HYPEROPT:

            logger.info(f"Optimizing hyperparameters for {model_type}...")

            if is_sequence:

                # Use sequence data for optimization

                best_model, best_params = optimize_with_optuna(

                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_type, is_sequence, n_trials=30

                )

            else:

                # Use non-sequence data for optimization

                best_model, best_params = optimize_with_optuna(

                    X_train_scaled, y_train, X_val_scaled, y_val, model_type, is_sequence, n_trials=30

                )

        else:

            # Simple grid search

            logger.info(f"Performing grid search for {model_type}...")

            if is_sequence:

                # Use sequence data for grid search

                best_model, best_params = hyperparameter_grid_search(

                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_type, is_sequence

                )

            else:

                # Use non-sequence data for grid search

                best_model, best_params = hyperparameter_grid_search(

                    X_train_scaled, y_train, X_val_scaled, y_val, model_type, is_sequence

                )

        # Build stacked ensemble

        logger.info("Building stacked ensemble model...")

        if is_sequence:

            # Sequence data - need to convert to flat for ensemble

            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, LOOKBACK)

            # Train the ensemble

            ensemble = build_stacked_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)

            # Evaluate ensemble on test set

            ensemble_pred = predict_with_stacked_ensemble(ensemble, X_test_scaled)

            ensemble_mse = mean_squared_error(y_test[LOOKBACK:], ensemble_pred[LOOKBACK:])

            ensemble_dir_acc = directional_accuracy_numpy(y_test[LOOKBACK:], ensemble_pred[LOOKBACK:])

            # Evaluate best model on test set

            model_pred = best_model.predict(X_test_seq).flatten()

            model_mse = mean_squared_error(y_test_seq, model_pred)

            model_dir_acc = directional_accuracy_numpy(y_test_seq, model_pred)

            logger.info(f"Ensemble - MSE: {ensemble_mse:.6f}, Dir Acc: {ensemble_dir_acc:.4f}")

            logger.info(f"Best Model - MSE: {model_mse:.6f}, Dir Acc: {model_dir_acc:.4f}")

            # Choose best between ensemble and single model

            if ensemble_dir_acc > model_dir_acc:

                logger.info("Ensemble model performs better, using it as final model")

                final_model = ensemble

                is_ensemble = True

                ensemble_models = [m for _, m in ensemble['base_models']]

                ensemble_weights = [1.0 / len(ensemble_models)] * len(ensemble_models)  # Equal weights for now

                # Re-run evaluation with ensemble for consistent outputs

                results = evaluate_model(

                    None, X_test_scaled, y_test[LOOKBACK:],

                    is_sequence=False, is_ensemble=True,

                    ensemble_models=ensemble_models, ensemble_weights=ensemble_weights

                )

            else:

                logger.info("Single model performs better, using it as final model")

                final_model = best_model

                is_ensemble = False

                ensemble_models = None

                ensemble_weights = None

                # Re-run evaluation with single model

                results = evaluate_model(best_model, X_test_seq, y_test_seq, is_sequence=True)

        else:

            # Non-sequence data

            # Train the ensemble

            ensemble = build_stacked_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)

            # Evaluate ensemble on test set

            ensemble_pred = predict_with_stacked_ensemble(ensemble, X_test_scaled)

            ensemble_mse = mean_squared_error(y_test, ensemble_pred)

            ensemble_dir_acc = directional_accuracy_numpy(y_test, ensemble_pred)

            # Evaluate best model on test set

            model_pred = best_model.predict(X_test_scaled)

            model_mse = mean_squared_error(y_test, model_pred)

            model_dir_acc = directional_accuracy_numpy(y_test, model_pred)

            logger.info(f"Ensemble - MSE: {ensemble_mse:.6f}, Dir Acc: {ensemble_dir_acc:.4f}")

            logger.info(f"Best Model - MSE: {model_mse:.6f}, Dir Acc: {model_dir_acc:.4f}")

            # Choose best between ensemble and single model

            if ensemble_dir_acc > model_dir_acc:

                logger.info("Ensemble model performs better, using it as final model")

                final_model = ensemble

                is_ensemble = True

                ensemble_models = [m for _, m in ensemble['base_models']]

                ensemble_weights = [1.0 / len(ensemble_models)] * len(ensemble_models)  # Equal weights

                # Re-run evaluation with ensemble for consistent outputs

                results = evaluate_model(

                    None, X_test_scaled, y_test,

                    is_sequence=False, is_ensemble=True,

                    ensemble_models=ensemble_models, ensemble_weights=ensemble_weights

                )

            else:

                logger.info("Single model performs better, using it as final model")

                final_model = best_model

                is_ensemble = False

                ensemble_models = None

                ensemble_weights = None

                # Re-run evaluation with single model

                results = evaluate_model(best_model, X_test_scaled, y_test, is_sequence=False)

        # Extract feature importance if using tree-based model

        if is_ensemble:

            # Get feature importance from the meta learner if it's a tree-based model

            meta_learner = ensemble['meta_learner']

            if hasattr(meta_learner, 'feature_importances_'):

                feature_importance = meta_learner.feature_importances_

                # Only keep importance for original features, not base model predictions

                base_model_count = len(ensemble['base_models'])

                feature_importance = feature_importance[base_model_count:]

                # Only use importance for the top 20 features used in meta learner

                if selected_indices is not None:

                    # Get the original feature names

                    selected_original_indices = selected_indices[:20]

                    feature_names_for_importance = [feature_list[i] for i in selected_original_indices]

                else:

                    feature_names_for_importance = feature_list[:20]

                # Plot and save feature importance

                top_features = plot_feature_importance(feature_importance, feature_names_for_importance)

                results['feature_importance'] = feature_importance

            else:

                # Use first base model's feature importance if meta learner doesn't have it

                for name, model in ensemble['base_models']:

                    if hasattr(model, 'feature_importances_'):

                        feature_importance = model.feature_importances_

                        if selected_indices is not None:

                            # Get the original feature names

                            feature_names_for_importance = [feature_list[i] for i in selected_indices]

                        else:

                            feature_names_for_importance = feature_list

                        # Plot and save feature importance

                        top_features = plot_feature_importance(feature_importance, feature_names_for_importance)

                        results['feature_importance'] = feature_importance

                        break

        else:

            # Single model feature importance

            if hasattr(final_model, 'feature_importances_'):

                feature_importance = final_model.feature_importances_

                if selected_indices is not None:

                    # Get the original feature names

                    feature_names_for_importance = [feature_list[i] for i in selected_indices]

                else:

                    feature_names_for_importance = feature_list

                # Plot and save feature importance

                top_features = plot_feature_importance(feature_importance, feature_names_for_importance)

                results['feature_importance'] = feature_importance

        # Calculate confidence scores for predictions

        logger.info("Calculating confidence scores for predictions...")

        if is_ensemble and ensemble_models:

            # Get individual model predictions

            individual_preds = []

            for model in ensemble_models:

                if is_sequence and not isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):

                    X_test_model_seq, _ = create_sequences(X_test_scaled, y_test, LOOKBACK)

                    pred = model.predict(X_test_model_seq).flatten()

                    individual_preds.append(pred)

                else:

                    pred = model.predict(X_test_scaled)

                    if isinstance(pred, np.ndarray) and len(pred.shape) > 1:
                        pred = pred.flatten()

                    individual_preds.append(pred)

            # Convert to numpy array

            individual_preds = np.array(individual_preds)

            # Calculate confidence based on ensemble agreement

            confidence_scores = calculate_confidence_scores(results['predictions'], individual_preds)

        else:

            # Single model confidence

            confidence_scores = calculate_confidence_scores(

                results['predictions'], model=final_model, X=X_test_scaled

            )

        # Create trading strategy with confidence filtering

        logger.info("Creating trading strategy with confidence filtering...")

        strategy_results = create_trading_strategy(test_df, results['predictions'], confidence_scores)

        # Save model and metadata

        logger.info("Saving model and metadata...")

        save_model_and_metadata(

            final_model if not is_ensemble else None,

            scaler,

            feature_list,

            selected_indices,

            best_params,

            results,

            is_ensemble=is_ensemble,

            ensemble_models=ensemble_models,

            ensemble_weights=ensemble_weights,

            day_models=day_models

        )

        # Save trading strategy results

        with open(os.path.join(RESULTS_DIR, 'trading_strategy.json'), 'w') as f:
            # Convert non-serializable parts
            strategy_data = {k: v for k, v in strategy_results.items() if k != 'strategy_df'}
            json.dump(make_json_serializable(strategy_data), f, indent=4)

        # Save strategy DataFrame

        strategy_results['strategy_df'].to_csv(os.path.join(RESULTS_DIR, 'strategy_results.csv'), index=False)

        # Create statistical significance tests

        y_test_actual = results['actual']

        y_test_pred = results['predictions']

        # Calculate correlation and p-value

        correlation, p_value = stats.pearsonr(y_test_actual, y_test_pred)

        statistical_tests = {

            'correlation': {

                'value': float(correlation),

                'p_value': float(p_value),

                'significant': p_value < 0.05

            }

        }

        # Save statistical tests

        with open(os.path.join(RESULTS_DIR, 'statistical_tests.json'), 'w') as f:
            json.dump(make_json_serializable(statistical_tests), f, indent=4)

        logger.info("Training completed successfully!")


    except Exception as e:

        logger.error(f"Error during training: {e}")

        import traceback

        logger.error(traceback.format_exc())


    finally:

        # Shutdown MT5 connection

        mt5.shutdown()

        logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()