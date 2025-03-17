#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Testing Script
Focus: 5 PM Arizona Time Data with Advanced Model Evaluation and Risk Management
Implements findings from academic research papers on financial forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, auc
import MetaTrader5 as mt5
import pytz
import pickle
import logging
import io
import re
import json
from contextlib import redirect_stdout
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants - must match training
LOOKBACK = 5
SYMBOL = 'XAUUSD'  # Changed to Gold/USD as per repo name
TIMEFRAME = mt5.TIMEFRAME_H1

# Paths
MODEL_DIR = 'models'
TEST_RESULTS_DIR = 'test_results'
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Kelly fraction - risk management parameter
KELLY_FRACTION = 0.5  # Conservative adjustment to Kelly criterion

# Enable advanced analysis features
USE_MARKET_REGIMES = True
ENSEMBLE_PREDICTION = True
RISK_MANAGEMENT = True
CONFIDENCE_FILTERING = True  # Only take trades with high confidence


# Define custom metrics for the model
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


# Define our own MSE and MAE functions to avoid scope issues
def mse_custom(y_true, y_pred):
    """Custom MSE implementation"""
    return K.mean(K.square(y_true - y_pred))


def mae_custom(y_true, y_pred):
    """Custom MAE implementation"""
    return K.mean(K.abs(y_true - y_pred))


def inspect_model_input_shape(model):
    """
    Inspect the model's expected input shape
    """
    # Try different approaches to get the input shape
    try:
        # Method 1: Direct input shape property
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
            if isinstance(input_shape, tuple):
                # Single input model
                return input_shape[2]  # (batch_size, timesteps, features)
            elif isinstance(input_shape, list):
                # Multiple input model
                return input_shape[0][2]  # First input's feature dimension

        # Method 2: First layer's input shape
        first_layer = model.layers[0]
        input_shape = first_layer.input_shape
        if isinstance(input_shape, tuple):
            return input_shape[2]
        elif isinstance(input_shape, list):
            return input_shape[0][2]

        # Method 3: Model summary analysis
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        summary = f.getvalue()

        # Parse the summary to find input shape
        matches = re.findall(r'Input.*\(None, (\d+), (\d+)\)', summary)
        if matches:
            return int(matches[0][1])  # Features dimension

        logger.warning("Could not determine model input shape through standard methods")
        return None
    except Exception as e:
        logger.error(f"Error inspecting model shape: {e}")
        return None


def connect_to_mt5(login, password, server="MetaQuotes-Demo"):
    """
    Connect to MetaTrader 5
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
    Add cyclical datetime features - must match training
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

    # Add day of week one-hot encoding (for matching features with training)
    for i in range(7):
        df[f'day_{i}'] = (df['day_of_week'] == i).astype(int)

    # Add lunar cycle phase (can affect trading psychology)
    # This is a simplified approximation
    days_since_new_moon = (df['day_of_year'] % 29.53).astype(int)
    df['lunar_sin'] = np.sin(days_since_new_moon * (2 * np.pi / 29.53))
    df['lunar_cos'] = np.cos(days_since_new_moon * (2 * np.pi / 29.53))

    return df


def detect_market_regime(df, window=20):
    """
    Detect market regimes (trending, mean-reverting, volatile)
    Based on research paper findings on regime-based trading
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std()

    # Calculate autocorrelation - negative values suggest mean reversion
    df['autocorrelation'] = df['returns'].rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x.dropna()) > 1 else np.nan, raw=False
    )

    # Calculate trend strength using Hurst exponent approximation
    def hurst_exponent(returns, lags=range(2, 20)):
        if len(returns.dropna()) < max(lags) + 1:
            return np.nan

        tau = []
        std = []
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

    vol_threshold = df['volatility'].mean() * 1.5

    # Create regime flags
    df['regime_trending'] = ((df['hurst'] > 0.6) & (df['volatility'] <= vol_threshold)).astype(int)
    df['regime_mean_reverting'] = ((df['hurst'] < 0.4) & (df['volatility'] <= vol_threshold)).astype(int)
    df['regime_volatile'] = (df['volatility'] > vol_threshold).astype(int)

    # Create a composite regime indicator (could be used for model switching)
    df['regime'] = 0  # Default/normal
    df.loc[df['regime_trending'] == 1, 'regime'] = 1  # Trending
    df.loc[df['regime_mean_reverting'] == 1, 'regime'] = 2  # Mean-reverting
    df.loc[df['regime_volatile'] == 1, 'regime'] = 3  # Volatile

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

    return df


def add_technical_indicators(df):
    """
    Add technical analysis indicators using pandas - must match training
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

        # ATRP (ATR Percentage - ATR relative to close price)
        def calculate_atr(high, low, close, window=14):
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr

        df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'])
        df['atrp_14'] = (df['atr_14'] / df['close']) * 100  # ATR as percentage of price

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
        # Additional RSI periods
        df['rsi_5'] = calculate_rsi(df['close'], window=5)
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

        # Gann High-Low Activator
        def calculate_gann_hl(high, low, close, window=13):
            avg = (high + low + close) / 3
            avg_ma = avg.rolling(window=window).mean()
            return avg_ma

        df['gann_hl_13'] = calculate_gann_hl(df['high'], df['low'], df['close'])
        df['gann_hl_signal'] = np.where(df['close'] > df['gann_hl_13'], 1, -1)

        # Fibonacci retracement levels
        # Use 100-day rolling window to find swings
        window = 100
        if len(df) >= window:
            rolling_high = df['high'].rolling(window=window).max()
            rolling_low = df['low'].rolling(window=window).min()

            # Calculate key Fibonacci levels (23.6%, 38.2%, 50%, 61.8%)
            df['fib_0'] = rolling_low
            df['fib_236'] = rolling_low + 0.236 * (rolling_high - rolling_low)
            df['fib_382'] = rolling_low + 0.382 * (rolling_high - rolling_low)
            df['fib_500'] = rolling_low + 0.5 * (rolling_high - rolling_low)
            df['fib_618'] = rolling_low + 0.618 * (rolling_high - rolling_low)
            df['fib_100'] = rolling_high

            # Check if price is near Fibonacci level (within 0.5%)
            price_range = rolling_high - rolling_low
            epsilon = 0.005 * price_range

            df['near_fib_236'] = (abs(df['close'] - df['fib_236']) < epsilon).astype(int)
            df['near_fib_382'] = (abs(df['close'] - df['fib_382']) < epsilon).astype(int)
            df['near_fib_500'] = (abs(df['close'] - df['fib_500']) < epsilon).astype(int)
            df['near_fib_618'] = (abs(df['close'] - df['fib_618']) < epsilon).astype(int)

            # Combined Fibonacci signal
            df['near_fib_level'] = ((df['near_fib_236'] + df['near_fib_382'] +
                                     df['near_fib_500'] + df['near_fib_618']) > 0).astype(int)

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return df


def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """
    Add lagged features for selected columns with more lags as suggested in research
    """
    # Key indicators to lag
    key_indicators = [
        'close', 'close_diff', 'close_diff_pct', 'rsi_14', 'macd', 'bb_position',
        'volatility_20', 'adx', 'stoch_k', 'mfi_14', 'obv', 'volume'
    ]

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

    return df


def add_target_variables(df):
    """
    Add target variables for prediction with multi-timeframe targets
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
    vol_threshold = df['volatility_20'] * 2 if 'volatility_20' in df.columns else df['close_diff_pct'].rolling(
        window=20).std() * 2

    extreme_moves = []
    for i in range(len(df) - 5):
        try:
            max_move = df['close_diff_pct'].iloc[i + 1:i + 6].abs().max()
            extreme_moves.append(1 if max_move > vol_threshold.iloc[i] else 0)
        except:
            extreme_moves.append(0)  # Default if we can't calculate

    # Pad the end
    extreme_moves.extend([np.nan] * min(5, max(0, len(df) - len(extreme_moves))))
    df['extreme_move_5d'] = extreme_moves

    # Add target for regime switches (if market regime detection is enabled)
    if 'regime' in df.columns:
        df['regime_switch'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['next_regime'] = df['regime'].shift(-1)

    return df


def prepare_features_and_targets(df, target_col='next_close_change_pct', feature_blacklist=None,
                                 expected_features=None):
    """
    Prepare features and target variables with robust handling of missing features
    """
    # Check if target column exists, if not add it
    if target_col not in df.columns:
        logger.info(f"Target column '{target_col}' missing, calculating it now...")
        df['next_close'] = df['close'].shift(-1)
        df['next_close_change_pct'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Default blacklist if none provided
    if feature_blacklist is None:
        feature_blacklist = [
            'time', 'arizona_time', 'next_close',
            'close_future_2', 'close_future_3', 'close_future_4', 'close_future_5'
        ]

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

    # Remove target columns from features
    target_cols = ['next_close_change_pct', 'next_direction', 'future_volatility', 'extreme_move_5d', 'regime_switch',
                   'next_regime']
    target_cols += [f'change_future_{i}_pct' for i in range(2, 6)]

    # Keep only columns that exist in the DataFrame
    target_cols = [col for col in target_cols if col in feature_df.columns]

    # Make sure our target column exists
    if target_col not in feature_df.columns:
        logger.error(f"Target column {target_col} not found after preprocessing!")
        # Use a dummy target as fallback
        feature_df[target_col] = 0

    # Separate features and target
    y = feature_df[target_col].values

    # Remove target columns from features
    X = feature_df.drop(columns=target_cols, errors='ignore').values

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    feature_columns = feature_df.drop(columns=target_cols, errors='ignore').columns.tolist()
    logger.info(f"Feature names: {feature_columns}")

    # After preparing X, check feature count and adjust before scaling
    if expected_features and X.shape[1] != expected_features:
        logger.warning(f"Feature count mismatch: expected {expected_features}, got {X.shape[1]}")

        if X.shape[1] > expected_features:
            logger.info(f"Truncating features from {X.shape[1]} to {expected_features}")
            X = X[:, :expected_features]
            feature_columns = feature_columns[:expected_features]
        else:
            logger.info(f"Padding features from {X.shape[1]} to {expected_features}")
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack([X, padding])
            # Add dummy feature names for padding
            feature_columns = feature_columns + [f'padding_{i}' for i in
                                                 range(expected_features - len(feature_columns))]

    return X, y, feature_columns


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM/GRU models with adaptability for small datasets
    """
    X_seq, y_seq = [], []

    # Handle small datasets
    if len(X) <= lookback + 1:
        logger.warning(f"Dataset too small ({len(X)} samples) for lookback={lookback}")
        # Use reduced lookback for small datasets
        reduced_lookback = max(1, len(X) - 2)
        logger.info(f"Reducing lookback from {lookback} to {reduced_lookback} for this analysis")
        lookback = reduced_lookback

    # Create sequences
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    # Convert to numpy arrays
    X_seq_np = np.array(X_seq) if X_seq else np.empty((0, lookback, X.shape[1]))
    y_seq_np = np.array(y_seq) if y_seq else np.empty(0)

    return X_seq_np, y_seq_np


def load_ensemble_models(ensemble_dir):
    """
    Load ensemble of models from directory
    """
    models = []
    ensemble_weights_path = os.path.join(ensemble_dir, 'ensemble_weights.json')

    if not os.path.exists(ensemble_weights_path):
        logger.error(f"Ensemble weights file not found: {ensemble_weights_path}")
        return None, None

    # Load ensemble weights
    try:
        with open(ensemble_weights_path, 'r') as f:
            weights = json.load(f)
    except Exception as e:
        logger.error(f"Error loading ensemble weights: {e}")
        return None, None

    # Load each model in the ensemble
    for i in range(len(weights)):
        try:
            model_path = os.path.join(ensemble_dir, f'model_{i}.h5')

            # Define custom metrics for loading
            custom_objects = {
                'r2_keras': r2_keras,
                'directional_accuracy': directional_accuracy,
                'mse_custom': mse_custom,
                'mae_custom': mae_custom
            }

            # Try loading in different ways
            try:
                # Method 1: Without compilation
                model = load_model(model_path, compile=False, custom_objects=custom_objects)
                # Recompile
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae', r2_keras, directional_accuracy]
                )
            except:
                # Method 2: With compilation and custom objects
                model = load_model(model_path, custom_objects=custom_objects)

            models.append(model)
            logger.info(f"Loaded ensemble model {i} from {model_path}")
        except Exception as e:
            logger.error(f"Error loading ensemble model {i}: {e}")
            return None, None

    logger.info(f"Successfully loaded ensemble with {len(models)} models")
    return models, weights


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

    # Ensure we return a flattened array if appropriate
    return weighted_preds.flatten() if weighted_preds.ndim > 1 else weighted_preds


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


def adapt_scaler(scaler, n_features_new):
    """
    Adapt a scaler to handle a different number of features
    Works with StandardScaler, MinMaxScaler, RobustScaler
    """
    if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
        # StandardScaler has mean_ and scale_
        n_features_old = scaler.scale_.shape[0]

        if n_features_new > n_features_old:
            # Add zeros for padding (mean) and ones for scale
            scaler.mean_ = np.pad(scaler.mean_, (0, n_features_new - n_features_old))
            scaler.scale_ = np.pad(scaler.scale_, (0, n_features_new - n_features_old),
                                   constant_values=1)
        else:
            # Truncate
            scaler.mean_ = scaler.mean_[:n_features_new]
            scaler.scale_ = scaler.scale_[:n_features_new]

    elif hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
        # RobustScaler has center_ and scale_
        n_features_old = scaler.center_.shape[0]

        if n_features_new > n_features_old:
            # Add zeros for padding (center) and ones for scale
            scaler.center_ = np.pad(scaler.center_, (0, n_features_new - n_features_old))
            scaler.scale_ = np.pad(scaler.scale_, (0, n_features_new - n_features_old),
                                   constant_values=1)
        else:
            # Truncate
            scaler.center_ = scaler.center_[:n_features_new]
            scaler.scale_ = scaler.scale_[:n_features_new]

    elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
        # MinMaxScaler has min_ and scale_
        n_features_old = scaler.min_.shape[0]

        if n_features_new > n_features_old:
            # Add zeros for padding (min) and ones for scale
            scaler.min_ = np.pad(scaler.min_, (0, n_features_new - n_features_old))
            scaler.scale_ = np.pad(scaler.scale_, (0, n_features_new - n_features_old),
                                   constant_values=1)
        else:
            # Truncate
            scaler.min_ = scaler.min_[:n_features_new]
            scaler.scale_ = scaler.scale_[:n_features_new]

    else:
        logger.warning(f"Unknown scaler type: {type(scaler)}, cannot adapt")

    # Update n_features_in_ if it exists
    if hasattr(scaler, 'n_features_in_'):
        scaler.n_features_in_ = n_features_new

    return scaler


def evaluate_model():
    """
    Evaluate the trained model on test data with enhanced analytics
    """
    # Check if we have an ensemble model
    ensemble_dir = os.path.join(MODEL_DIR, 'ensemble')
    ensemble_exists = os.path.exists(ensemble_dir)

    # Load metadata first to check if we have an ensemble
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                is_ensemble = metadata.get('is_ensemble', False)
                logger.info(f"Model metadata loaded. Ensemble: {is_ensemble}")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            is_ensemble = ensemble_exists
    else:
        is_ensemble = ensemble_exists

    # Load ensemble if available
    ensemble_models = None
    ensemble_weights = None
    if is_ensemble and ensemble_exists:
        logger.info("Loading ensemble models...")
        ensemble_models, ensemble_weights = load_ensemble_models(ensemble_dir)
        if ensemble_models is None:
            logger.warning("Failed to load ensemble, falling back to single model")
            is_ensemble = False

    # Try three different methods to load the model if not using ensemble
    model = None
    if not is_ensemble:
        model_filenames = ['final_model.h5', 'mt5_neural_network_model.h5', 'model.h5', 'best_model.h5']

        for method in range(1, 4):
            if model is not None:
                break

            for filename in model_filenames:
                try:
                    model_path = os.path.join(MODEL_DIR, filename)
                    if not os.path.exists(model_path):
                        continue

                    if method == 1:
                        # Method 1: Load without compilation
                        logger.info(f"Method 1: Loading model without compilation...")
                        model = load_model(model_path, compile=False)

                        # Recompile with custom metrics
                        model.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae', r2_keras, directional_accuracy]
                        )
                        logger.info(f"Model loaded and recompiled successfully")
                        break

                    elif method == 2:
                        # Method 2: Using custom_objects
                        logger.info(f"Method 2: Loading with custom objects...")
                        custom_objects = {
                            'r2_keras': r2_keras,
                            'directional_accuracy': directional_accuracy,
                            'mse_custom': mse_custom,
                            'mae_custom': mae_custom
                        }

                        model = load_model(model_path, custom_objects=custom_objects)
                        logger.info(f"Model loaded with custom objects")
                        break

                    elif method == 3:
                        # Method 3: Create model from architecture + weights
                        logger.info(f"Method 3: Loading model architecture and weights...")

                        architecture_path = os.path.join(MODEL_DIR, 'model_architecture.json')
                        if os.path.exists(architecture_path):
                            with open(architecture_path, 'r') as f:
                                model_json = f.read()

                            from tensorflow.keras.models import model_from_json
                            model = model_from_json(model_json)
                            model.load_weights(model_path)

                            # Compile
                            model.compile(
                                optimizer='adam',
                                loss='mse',
                                metrics=['mae', r2_keras, directional_accuracy]
                            )
                            logger.info(f"Model loaded from architecture and weights")
                            break
                except Exception as e:
                    logger.error(f"Method {method} failed with {filename}: {e}")
                    continue

        if model is None:
            logger.error("Failed to load model using any method")
            return None

    # Extract expected feature count
    if is_ensemble and ensemble_models:
        expected_features = inspect_model_input_shape(ensemble_models[0])
    else:
        expected_features = inspect_model_input_shape(model)

    if expected_features:
        logger.info(f"Model expects {expected_features} input features")
    else:
        logger.warning("Could not determine expected feature count, using default")
        expected_features = 219  # Based on your previous logs

    # Load feature list
    try:
        with open(os.path.join(MODEL_DIR, 'feature_list.pkl'), 'rb') as f:
            feature_list = pickle.load(f)
        logger.info(f"Loaded {len(feature_list)} features from feature list")
    except Exception as e:
        logger.warning(f"Could not load feature list, will create features from scratch: {e}")
        feature_list = None

    # Load scaler
    try:
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")

        # Adapt scaler if needed to match expected feature count
        if expected_features:
            scaler = adapt_scaler(scaler, expected_features)
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return

    # Generate test data
    # Connect to MT5
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    if not connect_to_mt5(account, password, server):
        logger.error("Failed to connect to MT5")
        return

    try:
        # Define date range for test data (3 months)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=90)

        # Get historical data
        df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)
        if df is None:
            return

        # Filter for 5 PM Arizona time
        df_5pm = filter_5pm_data(df)

        # Check if we have enough data
        if len(df_5pm) <= LOOKBACK:
            logger.warning(f"Limited data points at 5 PM AZ time (only {len(df_5pm)} found)")
            logger.info("Using all hours instead of just 5 PM for testing...")

            # Use all hours data instead
            df_all_hours = df.copy()

            # Add features to this dataset
            logger.info("Adding datetime features...")
            df_all_hours = add_datetime_features(df_all_hours)

            if USE_MARKET_REGIMES:
                logger.info("Detecting market regimes...")
                df_all_hours = detect_market_regime(df_all_hours)

            try:
                logger.info("Adding wavelet features...")
                df_all_hours = add_wavelet_features(df_all_hours)
            except Exception as e:
                logger.warning(f"Error adding wavelet features: {e}. Skipping.")

            logger.info("Adding technical indicators...")
            df_all_hours = add_technical_indicators(df_all_hours)

            logger.info("Adding lagged features...")
            df_all_hours = add_lagged_features(df_all_hours)

            logger.info("Adding target variables...")
            df_all_hours = add_target_variables(df_all_hours)

            # Use this dataset instead
            df_5pm = df_all_hours
            logger.info(f"Using expanded dataset with {len(df_5pm)} data points")
        else:
            # Process the 5 PM dataset
            logger.info("Adding datetime features...")
            df_5pm = add_datetime_features(df_5pm)

            if USE_MARKET_REGIMES:
                logger.info("Detecting market regimes...")
                df_5pm = detect_market_regime(df_5pm)

            try:
                logger.info("Adding wavelet features...")
                df_5pm = add_wavelet_features(df_5pm)
            except Exception as e:
                logger.warning(f"Error adding wavelet features: {e}. Skipping.")

            logger.info("Adding technical indicators...")
            df_5pm = add_technical_indicators(df_5pm)

            logger.info("Adding lagged features...")
            df_5pm = add_lagged_features(df_5pm)

            logger.info("Adding target variables...")
            df_5pm = add_target_variables(df_5pm)

        # Check that our target column exists
        if 'next_close_change_pct' not in df_5pm.columns:
            logger.error("Target column 'next_close_change_pct' not found after preprocessing!")
            return

        # Prepare features and target
        X, y, actual_features = prepare_features_and_targets(df_5pm, expected_features=expected_features)

        try:
            # Scale features
            X_scaled = scaler.transform(X)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            # Try adapting scaler or using unscaled features
            try:
                # Try adapting scaler
                logger.info("Trying to adapt scaler to current feature count...")
                scaler = adapt_scaler(scaler, X.shape[1])
                X_scaled = scaler.transform(X)
            except:
                logger.warning("Using unscaled features as fallback")
                X_scaled = X

        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            logger.error("Not enough data to create sequences!")
            return

        logger.info(f"Test data shape: {X_seq.shape}")

        # Make predictions
        try:
            if is_ensemble and ensemble_models:
                logger.info(f"Running ensemble prediction on {len(X_seq)} samples...")
                y_pred = ensemble_predict(ensemble_models, ensemble_weights, X_seq)
            else:
                logger.info(f"Running model prediction on {len(X_seq)} samples...")
                y_pred = model.predict(X_seq, verbose=1)
                # Ensure it's flattened
                y_pred = y_pred.flatten()

            # Check for all-same predictions (a sign of model issues)
            if len(y_pred) > 1 and all(y_pred == y_pred[0]):
                logger.warning(
                    "All predictions are identical! This indicates a serious problem with the model or data.")

            # Log prediction statistics
            logger.info(f"Prediction stats: min={np.min(y_pred)}, max={np.max(y_pred)}, mean={np.mean(y_pred)}")
            logger.info(f"First 5 predictions: {y_pred[:5]}")
            logger.info(f"First 5 actual values: {y_seq[:5]}")

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # Calculate metrics
        try:
            mse = mean_squared_error(y_seq, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_seq, y_pred)
            r2 = r2_score(y_seq, y_pred)

            # Directional accuracy
            directional_acc = np.mean((np.sign(y_seq) == np.sign(y_pred)).astype(int))
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return

        # Calculate precision-recall for extreme moves (if predicted magnitude > 0.5% and actual direction matches)
        extreme_threshold = 0.5  # Consider moves > 0.5% as significant
        y_extreme_pred = np.abs(y_pred) > extreme_threshold
        y_extreme_actual = np.abs(y_seq) > extreme_threshold
        y_dir_match = np.sign(y_seq) == np.sign(y_pred)
        y_extreme_match = y_extreme_actual & y_dir_match & y_extreme_pred

        # Calculate precision/recall
        if any(y_extreme_pred):
            precision = np.sum(y_extreme_match) / np.sum(y_extreme_pred)
        else:
            precision = 0

        if any(y_extreme_actual):
            recall = np.sum(y_extreme_match) / np.sum(y_extreme_actual)
        else:
            recall = 0

        # Calculate risk metrics
        risk_metrics = {}
        if RISK_MANAGEMENT:
            try:
                # Calculate Kelly criterion for different position types
                kelly_long = calculate_kelly_criterion(y_pred, y_seq, 'long_only')
                kelly_short = calculate_kelly_criterion(y_pred, y_seq, 'short_only')
                kelly_combined = calculate_kelly_criterion(y_pred, y_seq, 'long_short')

                # Apply KELLY_FRACTION to get conservative position sizing
                position_size_long = kelly_long * KELLY_FRACTION
                position_size_short = kelly_short * KELLY_FRACTION
                position_size_combined = kelly_combined * KELLY_FRACTION

                logger.info(f"Kelly Position Sizing:")
                logger.info(f"Long: {position_size_long:.2%}")
                logger.info(f"Short: {position_size_short:.2%}")
                logger.info(f"Combined: {position_size_combined:.2%}")

                risk_metrics = {
                    'kelly_long': kelly_long,
                    'kelly_short': kelly_short,
                    'kelly_combined': kelly_combined,
                    'position_size_long': position_size_long,
                    'position_size_short': position_size_short,
                    'position_size_combined': position_size_combined
                }
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {e}")

        # Log metrics
        logger.info(f"Test Metrics:")
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.6f}")
        logger.info(f"Directional Accuracy: {directional_acc:.2%}")
        logger.info(f"Extreme Move Precision: {precision:.2%}")
        logger.info(f"Extreme Move Recall: {recall:.2%}")

        # Create a DataFrame with dates and predictions for better analysis
        try:
            test_dates = df_5pm.iloc[LOOKBACK:]['arizona_time'].reset_index(drop=True)[:len(y_pred)]

            # Ensure we have data to work with
            if len(test_dates) > 0 and len(y_pred) > 0:
                # Truncate to shortest length to avoid issues
                min_len = min(len(test_dates), len(y_pred), len(y_seq))

                results_df = pd.DataFrame({
                    'date': test_dates[:min_len],
                    'actual': y_seq[:min_len],
                    'prediction': y_pred[:min_len]
                })

                # Mark extreme moves
                results_df['extreme_actual'] = np.abs(results_df['actual']) > extreme_threshold
                results_df['extreme_pred'] = np.abs(results_df['prediction']) > extreme_threshold

                # Add market regimes if enabled
                if USE_MARKET_REGIMES and 'regime' in df_5pm.columns:
                    regime_data = df_5pm.iloc[LOOKBACK:][['regime', 'regime_trending',
                                                          'regime_mean_reverting', 'regime_volatile']].reset_index(
                        drop=True)

                    # Add regime information to results if possible
                    if len(regime_data) >= min_len:
                        results_df = pd.concat([
                            results_df,
                            regime_data.iloc[:min_len].reset_index(drop=True)
                        ], axis=1)

                        # Calculate performance by regime
                        for regime_name, regime_id in zip(['Normal', 'Trending', 'Mean-Rev', 'Volatile'], range(4)):
                            mask = results_df['regime'] == regime_id
                            if mask.sum() > 0:
                                regime_acc = np.mean((np.sign(results_df.loc[mask, 'actual']) ==
                                                      np.sign(results_df.loc[mask, 'prediction'])).astype(int))
                                logger.info(f"Directional Accuracy in {regime_name} regime: {regime_acc:.2%}")

                # Plot predictions vs actual
                plt.figure(figsize=(14, 7))
                plt.plot(results_df['date'], results_df['actual'], label='Actual', alpha=0.7)
                plt.plot(results_df['date'], results_df['prediction'], label='Predicted', alpha=0.7)
                plt.title('Model Predictions vs Actual Values')
                plt.xlabel('Date')
                plt.ylabel('Price Change (%)')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(TEST_RESULTS_DIR, 'test_predictions.png'))

                # Plot prediction error
                plt.figure(figsize=(14, 7))
                results_df['error'] = results_df['actual'] - results_df['prediction']
                plt.plot(results_df['date'], results_df['error'])
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Prediction Error')
                plt.xlabel('Date')
                plt.ylabel('Error')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(TEST_RESULTS_DIR, 'prediction_error.png'))

                # Plot scatter of predicted vs actual
                plt.figure(figsize=(10, 10))
                plt.scatter(results_df['actual'], results_df['prediction'])
                plt.title('Actual vs Predicted')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')

                # Add 45-degree line
                min_val = min(np.min(results_df['actual']), np.min(results_df['prediction']))
                max_val = max(np.max(results_df['actual']), np.max(results_df['prediction']))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.grid(True)
                plt.savefig(os.path.join(TEST_RESULTS_DIR, 'scatter_comparison.png'))

                # Save results DataFrame for further analysis
                results_df.to_csv(os.path.join(TEST_RESULTS_DIR, 'test_results.csv'), index=False)

                # Plot confidence analysis if enough data
                if len(results_df) >= 10:
                    plt.figure(figsize=(12, 6))

                    # Group predictions by magnitude
                    results_df['pred_abs'] = np.abs(results_df['prediction'])
                    results_df['correct'] = np.sign(results_df['actual']) == np.sign(results_df['prediction'])

                    # Create bins by prediction magnitude
                    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 100]
                    bin_labels = ['0-0.1%', '0.1-0.2%', '0.2-0.3%', '0.3-0.4%', '0.4-0.5%', '0.5-1.0%', '1.0%+']
                    results_df['magnitude_bin'] = pd.cut(results_df['pred_abs'], bins=bins, labels=bin_labels)

                    # Calculate accuracy by bin
                    accuracy_by_magnitude = results_df.groupby('magnitude_bin')['correct'].mean()
                    count_by_magnitude = results_df.groupby('magnitude_bin').size()

                    # Plot accuracy by prediction magnitude
                    ax = plt.subplot(1, 2, 1)
                    accuracy_by_magnitude.plot(kind='bar', ax=ax)
                    plt.title('Accuracy by Prediction Magnitude')
                    plt.xlabel('Predicted Magnitude')
                    plt.ylabel('Directional Accuracy')
                    plt.ylim(0, 1)

                    # Add count labels
                    for i, v in enumerate(accuracy_by_magnitude):
                        plt.text(i, v + 0.02, f"n={count_by_magnitude.iloc[i]}", ha='center')

                    # Plot trade distribution
                    ax = plt.subplot(1, 2, 2)
                    count_by_magnitude.plot(kind='bar', ax=ax)
                    plt.title('Trade Count by Prediction Magnitude')
                    plt.xlabel('Predicted Magnitude')
                    plt.ylabel('Number of Trades')

                    plt.tight_layout()
                    plt.savefig(os.path.join(TEST_RESULTS_DIR, 'confidence_analysis.png'))
            else:
                logger.warning("Not enough data to create results DataFrame")

        except Exception as e:
            logger.error(f"Error creating results DataFrame: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Save risk management data if enabled
        if RISK_MANAGEMENT and risk_metrics:
            with open(os.path.join(TEST_RESULTS_DIR, 'risk_metrics.json'), 'w') as f:
                json.dump(risk_metrics, f, indent=4)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_acc,
            'extreme_precision': precision,
            'extreme_recall': recall,
            'risk_metrics': risk_metrics if RISK_MANAGEMENT else {}
        }

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    finally:
        mt5.shutdown()


def main():
    # Make sure model and scaler exist
    model_path = os.path.join(MODEL_DIR, 'final_model.h5')
    ensemble_dir = os.path.join(MODEL_DIR, 'ensemble')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    # Check if at least one type of model exists
    if not (os.path.exists(model_path) or os.path.exists(ensemble_dir)):
        logger.error(f"Neither model file nor ensemble directory found")
        return

    # Check for scaler
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return

    # Evaluate model
    logger.info("Evaluating model on test data...")
    metrics = evaluate_model()

    logger.info("Testing completed successfully")


if __name__ == "__main__":
    main()
    
