#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Validation Script
Focus: 5 PM Arizona Time Data with Advanced Model Evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MetaTrader5 as mt5
import pytz
import pickle
import logging
import json
import warnings
from collections import defaultdict
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

# Constants - must match training
LOOKBACK = 5  # Number of previous time periods to consider
SYMBOL = 'XAUUSD'  # Trading symbol - Gold/USD
TIMEFRAME = mt5.TIMEFRAME_H1  # 1-hour timeframe

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Define paths
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
RESULTS_DIR = 'results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Custom metrics for model loading
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


def ensure_numeric_data(X):
    """Ensure data has proper numeric dtype"""
    # Check if we have object dtype
    if X.dtype == 'object':
        logger.warning(f"Data has object dtype, converting to float32")

        # Try direct conversion first
        try:
            return X.astype(np.float32)
        except Exception as e:
            logger.warning(f"Direct conversion failed: {e}, trying column-by-column...")

        # If direct conversion fails, try column by column (for 2D arrays)
        if len(X.shape) == 2:
            X_numeric = np.zeros(X.shape, dtype=np.float32)
            for i in range(X.shape[1]):
                try:
                    X_numeric[:, i] = X[:, i].astype(np.float32)
                except Exception as e:
                    logger.warning(f"Column {i} conversion failed: {e}, using zeros")
            return X_numeric

        # For 3D arrays (sequences)
        elif len(X.shape) == 3:
            X_numeric = np.zeros(X.shape, dtype=np.float32)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        X_numeric[i, j] = X[i, j].astype(np.float32)
                    except:
                        logger.warning(f"Sequence {i}, timestep {j} conversion failed, using zeros")
            return X_numeric

    # If already numeric but not float32, convert
    if X.dtype != np.float32:
        return X.astype(np.float32)

    # Already float32
    return X


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

    # Special features for Wednesdays (which showed poor performance)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)

    # Market correlation features
    df['is_first_half_week'] = (df['day_of_week'] < 3).astype(int)
    df['is_second_half_week'] = (df['day_of_week'] >= 3).astype(int)

    # Special features for beginning/middle/end of month
    df['is_early_month'] = (df['day_of_month'] <= 10).astype(int)
    df['is_mid_month'] = ((df['day_of_month'] > 10) & (df['day_of_month'] <= 20)).astype(int)
    df['is_late_month'] = (df['day_of_month'] > 20).astype(int)

    return df


def detect_market_regime(df, window=20):
    """
    Detect market regimes (trending, mean-reverting, volatile)
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std()

    # Add realized volatility measures (standard, high frequency)
    df['realized_vol_10'] = df['returns'].rolling(window=10).apply(lambda x: np.sqrt(np.sum(x ** 2)) * np.sqrt(252))
    df['realized_vol_20'] = df['returns'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x ** 2)) * np.sqrt(252))

    # Parkinson volatility estimator (uses high-low range)
    df['high_low_ratio'] = df['high'] / df['low']
    df['log_high_low'] = np.log(df['high_low_ratio'])
    df['parkinsons_vol'] = df['log_high_low'].rolling(window=window).apply(
        lambda x: np.sqrt((1 / (4 * np.log(2))) * np.sum(x ** 2) / window)
    )

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

    # Trend strength indicators
    if 'adx' in df.columns:
        df['adx_trend'] = df['adx'] > 25

    # Volatility regime timing
    df['vol_expansion'] = (df['volatility'] > df['volatility'].shift(1)).astype(int)
    df['vol_contraction'] = (df['volatility'] < df['volatility'].shift(1)).astype(int)

    # Rate of change of volatility
    df['vol_roc'] = df['volatility'].pct_change() * 100

    # Volatility of volatility
    df['vol_of_vol'] = df['vol_roc'].rolling(window=window).std()

    return df


def add_wavelet_features(df, column='close', scales=[2, 4, 8, 16]):
    """
    Add wavelet transformation features
    Based on research paper findings on wavelet decomposition for better feature extraction
    Requires PyWavelets package
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
    except Exception as e:
        logger.warning(f"Error adding wavelet features: {e}")

    return df


def add_technical_indicators(df):
    """
    Add technical analysis indicators using pandas
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
            # Calculate intermediate WMAs
            half_length = int(window / 2)
            quarter_length = int(window / 4)

            # For smaller datasets, adjust calculations to prevent errors
            if len(df) >= half_length:
                df[f'wma_half_{window}'] = df['close'].rolling(window=half_length).apply(
                    lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                    raw=True
                )
            else:
                df[f'wma_half_{window}'] = df['close']

            if len(df) >= quarter_length:
                df[f'wma_quarter_{window}'] = df['close'].rolling(window=quarter_length).apply(
                    lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                    raw=True
                )
            else:
                df[f'wma_quarter_{window}'] = df['close']

            # Calculate HMA if enough data
            sqrt_window = int(np.sqrt(window))
            if len(df) >= sqrt_window:
                df[f'hma_{window}'] = df[f'wma_quarter_{window}'].rolling(window=sqrt_window).apply(
                    lambda x: sum((i + 1) * x[-(i + 1)] for i in range(len(x))) / sum(i + 1 for i in range(len(x))),
                    raw=True
                )
            else:
                df[f'hma_{window}'] = df[f'wma_quarter_{window}']

            # Clean up intermediate columns
            df = df.drop(columns=[f'wma_half_{window}', f'wma_quarter_{window}'])

        # Price relative to moving averages
        for window in [5, 10, 20, 50, 200]:
            if f'sma_{window}' in df.columns:
                df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            if f'ema_{window}' in df.columns:
                df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']

        # Moving average crossovers - important trading signals
        if 'sma_5' in df.columns and 'sma_10' in df.columns:
            df['sma_5_10_cross'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        if 'sma_10' in df.columns and 'sma_20' in df.columns:
            df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)  # Golden/Death cross
        if 'ema_5' in df.columns and 'ema_10' in df.columns:
            df['ema_5_10_cross'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
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

        # Limit the number of technical indicators to avoid memory issues
        # Add just a few more essential indicators

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

        # On Balance Volume (OBV) - simplified calculation
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        logger.error(traceback.format_exc())

    return df


def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """
    Add lagged features for selected columns
    """
    # Key indicators to lag - focus on the most important ones
    key_indicators = [
        'close', 'close_diff_pct', 'rsi_14', 'macd', 'bb_position',
        'volatility_20', 'stoch_k', 'obv', 'atr_14'
    ]

    # Add lags for all key indicators that exist in the dataframe
    for col in key_indicators:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Add rate of change between lags for close price
    for lag in lags[1:]:  # Skip the first lag
        if f'close_lag_{lag}' in df.columns and f'close_lag_1' in df.columns:
            df[f'close_lag_{lag}_1_diff'] = ((df[f'close_lag_1'] - df[f'close_lag_{lag}']) /
                                             df[f'close_lag_{lag}']) * 100

    return df


def add_target_variables(df):
    """
    Add target variables for prediction
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

    return df


def prepare_features_and_targets(df, target_col='next_close_change_pct', feature_blacklist=None,
                                 expected_features=None):
    """
    Prepare features and target variables with robust handling of missing features
    and explicit type conversion
    """
    # Check if target column exists
    if target_col not in df.columns:
        logger.info(f"Target column '{target_col}' missing, calculating it now...")
        df['next_close'] = df['close'].shift(-1)
        df['next_close_change_pct'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Default blacklist if none provided
    if feature_blacklist is None:
        feature_blacklist = [
            'time', 'arizona_time', 'date', 'next_close', 'hour',
            'close_future_2', 'close_future_3', 'close_future_4', 'close_future_5',
            'next_high', 'next_low'
        ]

    # Drop unnecessary columns
    drop_cols = feature_blacklist + [f'close_future_{i}' for i in range(2, 6)]
    feature_df = df.drop(columns=drop_cols, errors='ignore')

    # Check for object dtypes
    object_columns = feature_df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        logger.warning(f"Found {len(object_columns)} columns with object dtype")
        for col in object_columns:
            try:
                logger.info(f"Converting column {col} to numeric")
                feature_df[col] = feature_df[col].astype(float)
            except:
                logger.warning(f"Failed to convert column {col}, dropping it")
                feature_df = feature_df.drop(columns=[col])

    # Handle NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # For features, forward-fill then backward-fill
    feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')

    # Get remaining NaN columns and fill with zeros
    nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Filling these columns with zeros: {nan_cols}")
        feature_df[nan_cols] = feature_df[nan_cols].fillna(0)

    # Verify no more object dtypes exist
    object_columns = feature_df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        logger.warning(f"Still have object columns after conversion: {object_columns}")
        feature_df = feature_df.drop(columns=object_columns)

    # Separate features and target
    if target_col not in feature_df.columns:
        logger.error(f"Target column {target_col} not found after preprocessing!")
        logger.info(f"Available columns: {feature_df.columns.tolist()}")
        # Use a dummy target as fallback
        feature_df[target_col] = 0

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

    # Ensure numeric data types
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

    # Check for class imbalance in directional prediction
    if 'next_direction' in feature_df.columns:
        up_pct = feature_df['next_direction'].mean() * 100
        logger.info(f"Class balance - Up: {up_pct:.1f}%, Down: {100 - up_pct:.1f}%")

    # After preparing X, check feature count and adjust before scaling
    if expected_features is not None and expected_features > 0 and X.shape[1] != expected_features:
        logger.warning(f"Feature count mismatch: expected {expected_features}, got {X.shape[1]}")

        if X.shape[1] > expected_features:
            logger.info(f"Truncating features from {X.shape[1]} to {expected_features}")
            X = X[:, :expected_features]
            feature_names = feature_names[:expected_features]
        else:
            logger.info(f"Padding features from {X.shape[1]} to {expected_features}")
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=np.float32)
            X = np.hstack([X, padding])
            feature_names = feature_names + [f'padding_{i}' for i in range(expected_features - len(feature_names))]

    return X, y, feature_names


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences with fixed lookback regardless of dataset size
    """
    X_seq, y_seq = [], []

    # Don't reduce lookback for small datasets
    if len(X) <= lookback + 1:
        # Instead of reducing lookback, return empty arrays or skip this batch
        logger.warning(f"Dataset too small ({len(X)} samples) for lookback={lookback}, skipping")
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)

    # Create sequences with fixed lookback
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    # Convert to numpy arrays with explicit types
    X_seq_np = np.array(X_seq, dtype=np.float32)
    y_seq_np = np.array(y_seq, dtype=np.float32)

    return X_seq_np, y_seq_np


def scale_features(X, scaler=None):
    """
    Scale features using provided scaler or create a new one
    """
    try:
        if scaler is not None:
            # Use provided scaler
            X_scaled = scaler.transform(X)
        else:
            # Create a new scaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
    except Exception as e:
        logger.error(f"Error during scaling: {e}")
        logger.warning("Using unscaled data")
        X_scaled = X

    # Ensure numeric type
    X_scaled = ensure_numeric_data(X_scaled)

    return X_scaled, scaler


def load_model_and_scaler():
    """
    Load the trained model, scaler, and metadata
    """
    model = None
    scaler = None
    metadata = {}
    expected_features = None

    # Check if we have an ensemble model
    ensemble_dir = os.path.join(MODEL_DIR, 'ensemble')
    ensemble_exists = os.path.exists(ensemble_dir)
    is_ensemble = False
    ensemble_models = None
    ensemble_weights = None

    # Try to load metadata first
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                is_ensemble = metadata.get('is_ensemble', False)
                expected_features = metadata.get('selected_feature_count', None)
                logger.info(f"Model metadata loaded. Ensemble: {is_ensemble}, Expected features: {expected_features}")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            is_ensemble = ensemble_exists
    else:
        logger.warning("No metadata file found")
        is_ensemble = ensemble_exists

    # Load ensemble if available
    if is_ensemble and ensemble_exists:
        logger.info("Loading ensemble models...")
        ensemble_models = []

        # Try to load ensemble weights
        weights_path = os.path.join(ensemble_dir, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    ensemble_weights = json.load(f)
                    logger.info(f"Ensemble weights loaded: {ensemble_weights}")
            except Exception as e:
                logger.warning(f"Could not load ensemble weights: {e}")
                ensemble_weights = None

        # Set default weights if none were loaded
        if ensemble_weights is None:
            # Check how many models we have in the ensemble
            model_files = [f for f in os.listdir(ensemble_dir) if
                           f.startswith('model_') and (f.endswith('.h5') or f.endswith('.pkl'))]
            model_count = len(model_files)
            if model_count > 0:
                ensemble_weights = [1.0 / model_count] * model_count
                logger.info(f"Using equal weights for {model_count} ensemble models")
            else:
                logger.warning("No ensemble model files found")
                is_ensemble = False

        # Load individual models
        if is_ensemble:
            for i in range(len(ensemble_weights)):
                model_path = os.path.join(ensemble_dir, f'model_{i}.h5')
                if os.path.exists(model_path):
                    try:
                        # Define custom metrics for loading
                        custom_objects = {
                            'r2_keras': r2_keras,
                            'directional_accuracy': directional_accuracy,
                            'directional_weighted_loss': directional_weighted_loss
                        }

                        model = load_model(model_path, custom_objects=custom_objects)
                        ensemble_models.append(model)
                        logger.info(f"Loaded ensemble model {i}")
                    except Exception as e:
                        logger.error(f"Error loading ensemble model {i}: {e}")
                        continue

            if not ensemble_models:
                logger.warning("Failed to load any ensemble models, falling back to single model")
                is_ensemble = False

    # Load single model if not using ensemble
    if not is_ensemble:
        model_filenames = ['final_model.h5', 'mt5_neural_network_model.h5', 'model.h5', 'best_model.h5']

        # Try to load model with different approaches
        for filename in model_filenames:
            model_path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(model_path):
                continue

            try:
                # Try with custom objects
                custom_objects = {
                    'r2_keras': r2_keras,
                    'directional_accuracy': directional_accuracy,
                    'directional_weighted_loss': directional_weighted_loss
                }

                model = load_model(model_path, custom_objects=custom_objects)
                logger.info(f"Successfully loaded model from {model_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load model {filename} with custom objects: {e}")

                try:
                    # Try without compilation
                    model = load_model(model_path, compile=False)

                    # Recompile with custom metrics
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae', r2_keras, directional_accuracy]
                    )
                    logger.info(f"Successfully loaded and recompiled model from {model_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model {filename} without compilation: {e}")

    # If we still have no model, report error
    if not model and not ensemble_models:
        logger.error("Failed to load any model")
        return None, None, None, None, None

    # Try to get expected feature count from model
    if expected_features is None:
        try:
            if is_ensemble and ensemble_models:
                # Check first model's input shape
                first_layer = ensemble_models[0].layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = first_layer.input_shape
                    if isinstance(input_shape, tuple):
                        expected_features = input_shape[-1]
                    elif isinstance(input_shape, list):
                        expected_features = input_shape[0][-1]
            elif model:
                # Check model's input shape
                first_layer = model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = first_layer.input_shape
                    if isinstance(input_shape, tuple):
                        expected_features = input_shape[-1]
                    elif isinstance(input_shape, list):
                        expected_features = input_shape[0][-1]

            if expected_features:
                logger.info(f"Detected {expected_features} features required by model")
        except Exception as e:
            logger.warning(f"Could not determine expected feature count from model: {e}")

    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Successfully loaded scaler")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            scaler = None

    # Return all loaded components
    return model, ensemble_models, ensemble_weights, scaler, expected_features


def ensemble_predict(models, weights, X):
    """
    Make predictions with ensemble model using weighted average
    """
    if not models:
        logger.error("No models provided for ensemble prediction")
        return None

    predictions = []

    # Get predictions from each model
    for model in models:
        try:
            pred = model.predict(X, verbose=0).flatten()
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            # If prediction fails, add zeros as fallback
            predictions.append(np.zeros(len(X)))

    # Check if we have any valid predictions
    if not predictions:
        logger.error("No valid predictions from ensemble models")
        return np.zeros(len(X))

    # Convert to numpy array for calculations
    predictions = np.array(predictions)

    # Adjust weights if necessary
    if len(weights) != len(predictions):
        logger.warning(
            f"Weight count ({len(weights)}) doesn't match model count ({len(predictions)}), using equal weights")
        weights = [1.0 / len(predictions)] * len(predictions)

    # Calculate weighted average
    weighted_preds = np.zeros(predictions[0].shape)
    for i, pred in enumerate(predictions):
        weighted_preds += weights[i] * pred

    return weighted_preds


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using various metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = 0  # Default if calculation fails

    # Directional accuracy
    dir_acc = directional_accuracy_numpy(y_true, y_pred)

    # Create metrics dictionary
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': dir_acc
    }

    # Log metrics
    logger.info(f"Metrics:")
    logger.info(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}, R²: {r2:.6f}")
    logger.info(f"Directional Accuracy: {dir_acc:.4f}")

    return metrics


def walk_forward_validation(df, model, scaler, is_ensemble=False, ensemble_models=None, ensemble_weights=None,
                            expected_features=None):
    """
    Perform walk-forward validation with expanding window approach
    """
    # Ensure we have enough data
    min_train_size = 30  # Minimum number of samples for initial training

    if len(df) < min_train_size + 10:
        logger.error(f"Not enough data for walk-forward validation. Need at least {min_train_size + 10} samples")
        return None, None

    # Prepare for walk-forward validation
    all_predictions = []
    all_actuals = []
    validation_start = min_train_size

    # Create step size - validate every N rows
    step_size = max(5, LOOKBACK + 2)   # Validate every 5 rows for efficiency

    # Loop through validation points
    for i in range(validation_start, len(df), step_size):
        # Get training and validation data
        train_df = df.iloc[:i].copy()
        test_df = df.iloc[i:i + step_size].copy()

        if len(test_df) == 0:
            continue

        # Extract features and targets
        X_train, y_train, feature_names = prepare_features_and_targets(train_df, expected_features=expected_features)
        X_test, y_test, _ = prepare_features_and_targets(test_df, expected_features=expected_features)

        # Scale features
        try:
            X_train_scaled = scaler.transform(X_train)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            # Use unscaled data as fallback
            X_train_scaled = ensure_numeric_data(X_train)

        try:
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            logger.error(f"Error scaling test features: {e}")
            # Use unscaled data as fallback
            X_test_scaled = ensure_numeric_data(X_test)

        # Ensure numeric data
        X_train_scaled = ensure_numeric_data(X_train_scaled)
        X_test_scaled = ensure_numeric_data(X_test_scaled)

        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)

        # Add this check
        if X_test_seq.shape[0] > 0 and X_test_seq.shape[2] != 219:  # Check if feature count matches
            if X_test_seq.shape[2] < 219:
                # Pad features if needed
                padding = np.zeros((X_test_seq.shape[0], X_test_seq.shape[1], 219 - X_test_seq.shape[2]),
                                   dtype=np.float32)
                X_test_seq = np.concatenate([X_test_seq, padding], axis=2)
                logger.info(f"Padded sequence features from {X_test_seq.shape[2]} to 219")
            else:
                # Truncate features if needed
                X_test_seq = X_test_seq[:, :, :219]
                logger.info(f"Truncated sequence features to 219")

        # Skip if we don't have enough test data for sequences
        if len(X_test_seq) == 0:
            continue

        # Ensure sequences are numeric
        X_test_seq = ensure_numeric_data(X_test_seq)

        # Make predictions
        try:
            if is_ensemble and ensemble_models:
                predictions = ensemble_predict(ensemble_models, ensemble_weights, X_test_seq)
            else:
                # Ensure X_test_seq is still float32
                if X_test_seq.dtype != np.float32:
                    logger.warning(f"X_test_seq dtype is {X_test_seq.dtype}, converting to float32")
                    X_test_seq = X_test_seq.astype(np.float32)

                    expected_shape = (None, 5, 219)
                    if X_test_seq.shape[1:] != expected_shape[1:]:
                        logger.error(f"Shape mismatch: expected {expected_shape}, got {X_test_seq.shape}")
                        continue  # Skip this validation step

                predictions = model.predict(X_test_seq, verbose=0)
                predictions = predictions.flatten()
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            logger.error(traceback.format_exc())
            continue

        # Collect predictions and actuals
        all_predictions.extend(predictions)
        all_actuals.extend(y_test_seq)

    # Calculate overall metrics
    metrics = None
    if all_predictions and all_actuals:
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        metrics = evaluate_predictions(all_actuals, all_predictions)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'actual': all_actuals,
        'predicted': all_predictions
    })

    return results_df, metrics


def main():
    """Main validation function"""
    try:
        # Load model, scaler, and metadata
        logger.info("Loading model, scaler, and metadata...")
        model, ensemble_models, ensemble_weights, scaler, expected_features = load_model_and_scaler()

        if (model is None and ensemble_models is None) or scaler is None:
            logger.error("Failed to load model or scaler")
            return

        # MT5 connection params
        account = 90933473
        password = "NhXgR*3g"
        server = "MetaQuotes-Demo"

        # Connect to MT5
        if not connect_to_mt5(account, password, server):
            return

        try:
            # Define date range for historical data
            end_date = datetime.now(ARIZONA_TZ)
            start_date = end_date - timedelta(days=365)  # 1 year of data

            # Get historical data
            logger.info(f"Fetching historical data from {start_date} to {end_date}")
            df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)

            if df is None or len(df) == 0:
                logger.error("Failed to get historical data")
                return

            # Filter for 5 PM Arizona time
            df_5pm = filter_5pm_data(df)

            # Check if we have enough 5 PM data
            if len(df_5pm) < 30:
                logger.warning(f"Not enough 5 PM data (only {len(df_5pm)} samples). Using all hours data.")
                df_processed = df.copy()
            else:
                df_processed = df_5pm.copy()

            # Process the data for validation
            logger.info("Processing data for validation...")
            logger.info("Adding datetime features...")
            df_processed = add_datetime_features(df_processed)

            logger.info("Adding technical indicators...")
            df_processed = add_technical_indicators(df_processed)

            logger.info("Adding lagged features...")
            df_processed = add_lagged_features(df_processed)

            logger.info("Adding target variables...")
            df_processed = add_target_variables(df_processed)

            # Check if we have enough data after processing
            if len(df_processed) < 30:
                logger.error(f"Not enough data after processing (only {len(df_processed)} samples)")
                return

            # Determine if using ensemble
            is_ensemble = ensemble_models is not None and len(ensemble_models) > 0
            logger.info(f"Using {'ensemble' if is_ensemble else 'single'} model for validation")

            # Run walk-forward validation
            logger.info("Starting walk-forward validation...")
            wf_results, wf_metrics = walk_forward_validation(
                df_processed,
                model,
                scaler,
                is_ensemble=is_ensemble,
                ensemble_models=ensemble_models,
                ensemble_weights=ensemble_weights,
                expected_features=expected_features
            )

            if wf_results is None or wf_metrics is None:
                logger.error("Walk-forward validation failed")
                return

            # Save validation results
            os.makedirs(RESULTS_DIR, exist_ok=True)

            # Save metrics as JSON
            with open(os.path.join(RESULTS_DIR, 'validation_metrics.json'), 'w') as f:
                json.dump(wf_metrics, f, indent=4)

            # Save results DataFrame
            wf_results.to_csv(os.path.join(RESULTS_DIR, 'validation_results.csv'), index=False)

            # Plot validation results
            plt.figure(figsize=(14, 7))
            plt.plot(wf_results.index, wf_results['actual'], label='Actual', alpha=0.7)
            plt.plot(wf_results.index, wf_results['predicted'], label='Predicted', alpha=0.7)
            plt.title('Walk-Forward Validation - Predictions vs Actual')
            plt.xlabel('Sample')
            plt.ylabel('Price Change (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'validation_plot.png'))

            # Plot prediction error
            plt.figure(figsize=(14, 7))
            wf_results['error'] = wf_results['actual'] - wf_results['predicted']
            plt.plot(wf_results.index, wf_results['error'])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Prediction Error')
            plt.xlabel('Sample')
            plt.ylabel('Error')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'error_plot.png'))

            # Plot scatter of predicted vs actual
            plt.figure(figsize=(10, 10))
            plt.scatter(wf_results['actual'], wf_results['predicted'], alpha=0.5)
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')

            # Add 45-degree line
            min_val = min(wf_results['actual'].min(), wf_results['predicted'].min())
            max_val = max(wf_results['actual'].max(), wf_results['predicted'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, 'scatter_plot.png'))

            logger.info("Validation complete!")
            logger.info(f"Directional accuracy: {wf_metrics.get('directional_accuracy', 0):.4f}")
            logger.info(f"MSE: {wf_metrics.get('mse', 0):.6f}")
            logger.info(f"R²: {wf_metrics.get('r2', 0):.6f}")

        except Exception as e:
            logger.error(f"Error during validation process: {e}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())

    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()
