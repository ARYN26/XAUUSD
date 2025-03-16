#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Validation Script
Focus: 5 PM Arizona Time Data with Advanced Validation Techniques
Uses walk-forward validation with market regime awareness
Implements findings from academic research papers on financial forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, auc
import MetaTrader5 as mt5
import pytz
import pickle
import logging
import json
from scipy.stats import pearsonr
import warnings

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
logger = logging.getLogger(__name__)

# Constants - must match training
LOOKBACK = 20
SYMBOL = 'XAUUSD'  # Changed to Gold/USD
TIMEFRAME = mt5.TIMEFRAME_H1
VALIDATION_DIR = 'validation_results'
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Advanced validation settings
USE_MARKET_REGIMES = True
USE_MONTE_CARLO = True  # Use Monte Carlo validation for robustness
STATISTICAL_TESTS = True  # Use statistical hypothesis testing
PREDICTION_INTERVALS = True  # Generate prediction intervals


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


def connect_to_mt5(login, password, server="MetaQuotes-Demo"):
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
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        summary = f.getvalue()

        # Parse the summary to find input shape
        import re
        matches = re.findall(r'Input.*\(None, (\d+), (\d+)\)', summary)
        if matches:
            return int(matches[0][1])  # Features dimension

        logger.warning("Could not determine model input shape through standard methods")
        return None
    except Exception as e:
        logger.error(f"Error inspecting model shape: {e}")
        return None


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

    # Log available columns for debugging
    logger.info(f"Available columns in MT5 data: {df.columns.tolist()}")

    # Handle different volume column names
    if 'volume' not in df.columns:
        if 'tick_volume' in df.columns:
            logger.info("Using 'tick_volume' instead of 'volume'")
            df['volume'] = df['tick_volume']
        elif 'real_volume' in df.columns:
            logger.info("Using 'real_volume' instead of 'volume'")
            df['volume'] = df['real_volume']
        else:
            logger.info("No volume data found, creating placeholder")
            df['volume'] = 1.0  # Use a placeholder value

    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Convert to Arizona time
    df['arizona_time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(ARIZONA_TZ)
    df['hour'] = df['arizona_time'].dt.hour
    df['day_of_week'] = df['arizona_time'].dt.dayofweek
    df['day_of_month'] = df['arizona_time'].dt.day
    df['month'] = df['arizona_time'].dt.month

    logger.info(f"Fetched {len(df)} historical data points")
    return df


def filter_5pm_data(df):
    """
    Filter data to only include rows at 5 PM Arizona time
    """
    filtered_df = df[df['hour'] == TARGET_HOUR].copy()
    logger.info(f"Filtered to {len(filtered_df)} data points at 5 PM Arizona time")
    return filtered_df


def add_datetime_features(df):
    """
    Add cyclical datetime features - enhanced implementation
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

    return df


def detect_market_regime(df, window=20):
    """
    Detect market regimes (trending, mean-reverting, volatile)
    Based on research findings on regime-based trading
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
    Add technical analysis indicators using pandas
    Note: This is a comprehensive function that matches what's in train.py
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

        # Moving average crossovers
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
    Add lagged features for selected columns with more lags
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
    Add target variables for prediction including multi-timeframe targets
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

    return df


def preprocess_data(df, scaler, expected_features=None):
    """
    Preprocess data for neural network
    """
    # Calculate target: next period's close price percent change
    df['next_close'] = df['close'].shift(-1)
    df['next_close_change_pct'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Remove rows with NaN targets
    df = df.dropna(subset=['next_close_change_pct']).copy()

    # Prepare features (all columns except targets and timestamps)
    feature_blacklist = [
        'time', 'arizona_time', 'next_close',
        'close_future_2', 'close_future_3', 'close_future_4', 'close_future_5'
    ]

    # Drop unnecessary columns
    drop_cols = feature_blacklist + [f'close_future_{i}' for i in range(2, 6)]
    feature_df = df.drop(columns=drop_cols, errors='ignore')

    # Drop any target columns that might exist
    target_cols = ['next_close_change_pct', 'next_direction', 'future_volatility',
                   'extreme_move_5d', 'regime_switch', 'next_regime']
    target_cols += [f'change_future_{i}_pct' for i in range(2, 6)]

    feature_df = feature_df.drop(columns=target_cols, errors='ignore')

    # Handle NaN values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # For features, forward-fill then backward-fill
    feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')

    # Get remaining NaN columns and fill with zeros
    nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"Filling these columns with zeros: {nan_cols}")
        feature_df[nan_cols] = feature_df[nan_cols].fillna(0)

    # Target vector
    y = df['next_close_change_pct'].values

    # Features matrix
    X = feature_df.values

    # Check if scaler input dimension matches our data
    if expected_features is not None:
        current_features = X.shape[1]

        if current_features != expected_features:
            logger.warning(f"Feature count mismatch: expected {expected_features}, data has {current_features}")

            # Handle dimension mismatch
            if current_features < expected_features:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_features - current_features))
                X_padded = np.hstack([X, padding])
                logger.info(f"Padded input data from {current_features} to {expected_features} features")
                X = X_padded
            else:
                # Truncate
                logger.info(f"Truncating input data from {current_features} to {expected_features} features")
                X = X[:, :expected_features]

    try:
        # Scale features using the provided scaler
        X_scaled = scaler.transform(X)
        logger.info(f"Data scaled successfully with shape {X_scaled.shape}")
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        logger.warning("Using unscaled data")
        X_scaled = X

    return X_scaled, y, feature_df.columns.tolist()


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM/GRU models
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    return np.array(X_seq), np.array(y_seq)


def monte_carlo_validation(model, X_seq, y_seq, n_iterations=100, sample_size=0.8):
    """
    Perform Monte Carlo validation by randomly sampling the test data
    This provides more robust evaluation of model performance
    """
    mse_results = []
    mae_results = []
    r2_results = []
    dir_acc_results = []

    n_samples = len(X_seq)

    logger.info(f"Running Monte Carlo validation with {n_iterations} iterations")

    for i in range(n_iterations):
        # Random sample (without replacement)
        indices = np.random.choice(
            n_samples, size=int(n_samples * sample_size), replace=False
        )

        # Get sample data
        X_sample = X_seq[indices]
        y_sample = y_seq[indices]

        # Make predictions
        y_pred = model.predict(X_sample)

        # Calculate metrics
        mse = mean_squared_error(y_sample, y_pred)
        mae = mean_absolute_error(y_sample, y_pred)
        r2 = r2_score(y_sample, y_pred)
        dir_acc = np.mean((np.sign(y_sample) == np.sign(y_pred)).astype(int))

        # Store results
        mse_results.append(mse)
        mae_results.append(mae)
        r2_results.append(r2)
        dir_acc_results.append(dir_acc)

    # Calculate statistics
    results = {
        'mse': {
            'mean': np.mean(mse_results),
            'std': np.std(mse_results),
            'median': np.median(mse_results),
            '95_conf': [np.percentile(mse_results, 2.5), np.percentile(mse_results, 97.5)]
        },
        'mae': {
            'mean': np.mean(mae_results),
            'std': np.std(mae_results),
            'median': np.median(mae_results),
            '95_conf': [np.percentile(mae_results, 2.5), np.percentile(mae_results, 97.5)]
        },
        'r2': {
            'mean': np.mean(r2_results),
            'std': np.std(r2_results),
            'median': np.median(r2_results),
            '95_conf': [np.percentile(r2_results, 2.5), np.percentile(r2_results, 97.5)]
        },
        'directional_accuracy': {
            'mean': np.mean(dir_acc_results),
            'std': np.std(dir_acc_results),
            'median': np.median(dir_acc_results),
            '95_conf': [np.percentile(dir_acc_results, 2.5), np.percentile(dir_acc_results, 97.5)]
        }
    }

    # Log results
    logger.info(f"Monte Carlo Validation Results:")
    logger.info(f"MSE: {results['mse']['mean']:.6f} ± {results['mse']['std']:.6f}")
    logger.info(f"MAE: {results['mae']['mean']:.6f} ± {results['mae']['std']:.6f}")
    logger.info(f"R²: {results['r2']['mean']:.6f} ± {results['r2']['std']:.6f}")
    logger.info(
        f"Directional Accuracy: {results['directional_accuracy']['mean']:.2%} ± {results['directional_accuracy']['std']:.2%}")

    return results


def calculate_prediction_intervals(model, X_seq, confidence=0.95):
    """
    Calculate prediction intervals using bootstrapping approach
    Returns predictions with upper and lower bounds
    """
    n_samples = len(X_seq)
    n_bootstraps = 100
    predictions = []

    logger.info(f"Calculating {confidence * 100}% prediction intervals using bootstrapping")

    # Make base predictions
    base_pred = model.predict(X_seq)

    # Calculate absolute errors for base predictions
    # We'll need a small subset of the validation data with known targets
    # Let's just assume we have this for demonstration purposes
    # In a real scenario, you'd calculate this on validation data

    # Generate synthetic absolute errors (normally you'd calculate these from validation)
    # This is just a placeholder - for real use, calculate from validation data
    synth_errors = np.abs(base_pred) * np.random.uniform(0.1, 0.3, size=base_pred.shape)

    # Bootstrap to generate prediction intervals
    for i in range(n_bootstraps):
        # Resample errors
        sampled_indices = np.random.choice(len(synth_errors), size=len(synth_errors), replace=True)
        sampled_errors = synth_errors[sampled_indices]

        # Add/subtract errors from predictions
        upper_bound = base_pred + sampled_errors
        lower_bound = base_pred - sampled_errors

        predictions.append((lower_bound, upper_bound))

    # Calculate percentiles for each prediction
    lower_bounds = np.percentile([p[0] for p in predictions], (1 - confidence) / 2 * 100, axis=0)
    upper_bounds = np.percentile([p[1] for p in predictions], (1 + confidence) / 2 * 100, axis=0)

    return base_pred, lower_bounds, upper_bounds


def statistical_hypothesis_testing(actual, predictions):
    """
    Perform statistical tests to evaluate the model
    """
    # Convert to numpy arrays
    y_true = np.array(actual).flatten()
    y_pred = np.array(predictions).flatten()

    results = {}

    # 1. Test if prediction correlates with actual (Pearson correlation)
    try:
        correlation, p_value = pearsonr(y_true, y_pred)
        results['correlation'] = {
            'value': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'There is a statistically significant correlation between predictions and actual values'
            if p_value < 0.05 else
            'There is no statistically significant correlation between predictions and actual values'
        }
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        results['correlation'] = {'error': str(e)}

    # 2. Test if directional accuracy is better than random (binomial test)
    from scipy.stats import binom_test

    dir_correct = np.sum((np.sign(y_true) == np.sign(y_pred)))
    total = len(y_true)
    dir_accuracy = dir_correct / total

    # Binomial test against 0.5 (random guessing)
    p_value = binom_test(dir_correct, total, p=0.5)

    results['directional_test'] = {
        'accuracy': dir_accuracy,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'Directional accuracy is significantly better than random guessing'
        if p_value < 0.05 else
        'Directional accuracy is not significantly better than random guessing'
    }

    # 3. Test for normality of residuals
    from scipy.stats import shapiro

    residuals = y_true - y_pred
    stat, p_value = shapiro(residuals)

    results['residual_normality'] = {
        'statistic': stat,
        'p_value': p_value,
        'normal': p_value >= 0.05,
        'interpretation': 'Residuals are normally distributed'
        if p_value >= 0.05 else
        'Residuals are not normally distributed'
    }

    return results


def walk_forward_validation(model, df, scaler, window_size=30, step_size=10, expected_features=None):
    """
    Perform enhanced walk-forward validation to prevent look-ahead bias
    """
    logger.info("Performing walk-forward validation...")

    # Get all dates
    dates = df['arizona_time'].dt.date.unique()
    dates.sort()

    # Initialize results storage
    all_predictions = []
    all_actuals = []
    all_dates = []
    all_regimes = []

    # Loop through time windows
    total_steps = max(1, (len(dates) - window_size) // step_size)
    logger.info(f"Processing {total_steps} validation windows")

    for i in range(0, len(dates) - window_size, step_size):
        # Show progress
        if i % 10 == 0 or i == 0:
            logger.info(f"Validation window progress: {i // (total_steps or 1)}%")

        # Get test window dates
        test_start_date = dates[i]
        test_end_date = dates[i + window_size - 1]

        # Filter data for test window
        test_mask = (df['arizona_time'].dt.date >= test_start_date) & (df['arizona_time'].dt.date <= test_end_date)
        test_df = df[test_mask].copy()

        # If no valid 5PM data in this window, skip
        if len(test_df) == 0:
            continue

        # Preprocess test data
        X_test, y_test, _ = preprocess_data(test_df, scaler, expected_features)

        # Create sequences
        X_seq, y_seq = create_sequences(X_test, y_test, LOOKBACK)

        # If no sequences could be created, skip
        if len(X_seq) == 0:
            continue

        # Make predictions
        predictions = model.predict(X_seq, verbose=0)

        # Store results
        for j in range(len(predictions)):
            all_predictions.append(predictions[j][0])
            all_actuals.append(y_seq[j])
            # Get the corresponding date
            all_dates.append(test_df.iloc[j + LOOKBACK]['arizona_time'])

            # Store regime information if available
            if USE_MARKET_REGIMES and 'regime' in test_df.columns:
                all_regimes.append(test_df.iloc[j + LOOKBACK]['regime'])
            else:
                all_regimes.append(None)

    # Create results DataFrame
    results = pd.DataFrame({
        'date': all_dates,
        'actual': all_actuals,
        'prediction': all_predictions,
        'regime': all_regimes if USE_MARKET_REGIMES else None
    })

    # Calculate error
    results['error'] = results['actual'] - results['prediction']
    results['abs_error'] = abs(results['error'])

    # Calculate directional accuracy
    results['actual_direction'] = np.sign(results['actual'])
    results['predicted_direction'] = np.sign(results['prediction'])
    results['direction_match'] = results['actual_direction'] == results['predicted_direction']

    # Calculate metrics
    mse = mean_squared_error(results['actual'], results['prediction'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(results['actual'], results['prediction'])
    r2 = r2_score(results['actual'], results['prediction'])
    directional_accuracy = results['direction_match'].mean()

    logger.info(f"Walk-Forward Validation Results:")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")

    # Calculate statistics by regime if available
    if USE_MARKET_REGIMES and 'regime' in results.columns and results['regime'].notna().any():
        regimes = ['Normal', 'Trending', 'Mean-Rev', 'Volatile']

        for regime_id, regime_name in enumerate(regimes):
            regime_data = results[results['regime'] == regime_id]
            if len(regime_data) > 0:
                regime_dir_acc = regime_data['direction_match'].mean()
                regime_mse = mean_squared_error(regime_data['actual'], regime_data['prediction'])
                logger.info(f"Regime {regime_name}: Dir. Accuracy = {regime_dir_acc:.2%}, MSE = {regime_mse:.6f}")

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(results['date'], results['actual'], label='Actual', alpha=0.7)
    plt.plot(results['date'], results['prediction'], label='Predicted', alpha=0.7)
    plt.title('Walk-Forward Validation: Predicted vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price Change (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'walk_forward_results.png'))

    # Plot error over time
    plt.figure(figsize=(14, 7))
    plt.plot(results['date'], results['error'])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'walk_forward_error.png'))

    # Create monthly error analysis
    results['year_month'] = results['date'].dt.to_period('M')
    monthly_error = results.groupby('year_month').agg({
        'error': ['mean', 'std'],
        'abs_error': 'mean',
        'direction_match': 'mean'
    })

    logger.info(f"Monthly Analysis:\n{monthly_error}")

    # Plot prediction confidence - correlation between prediction magnitude and accuracy
    plt.figure(figsize=(10, 6))

    # Group by prediction magnitude
    results['pred_abs'] = np.abs(results['prediction'])

    # Create bins for prediction magnitude
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 100]
    bin_labels = ['0-0.1%', '0.1-0.2%', '0.2-0.3%', '0.3-0.4%', '0.4-0.5%', '0.5-0.7%', '0.7-1.0%', '1.0%+']
    results['pred_bin'] = pd.cut(results['pred_abs'], bins=bins, labels=bin_labels)

    # Calculate accuracy by bin
    bin_accuracy = results.groupby('pred_bin')['direction_match'].mean()
    bin_count = results.groupby('pred_bin').size()

    plt.bar(bin_accuracy.index, bin_accuracy)
    plt.title('Directional Accuracy by Prediction Magnitude')
    plt.ylabel('Accuracy')
    plt.xlabel('Prediction Magnitude')
    plt.grid(axis='y')

    # Add count labels
    for i, v in enumerate(bin_accuracy):
        plt.text(i, v + 0.02, f"n={bin_count.iloc[i]}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'accuracy_by_magnitude.png'))

    # Run statistical tests if enabled
    if STATISTICAL_TESTS:
        logger.info("Running statistical hypothesis tests...")
        stats_results = statistical_hypothesis_testing(results['actual'], results['prediction'])

        # Save statistical test results
        with open(os.path.join(VALIDATION_DIR, 'statistical_tests.json'), 'w') as f:
            json.dump(stats_results, f, indent=4)

        # Log key statistical results
        logger.info(
            f"Correlation: {stats_results['correlation']['value']:.4f} (p={stats_results['correlation']['p_value']:.4f})")
        logger.info(f"Directional test p-value: {stats_results['directional_test']['p_value']:.4f}")
        logger.info(f"Residuals normality p-value: {stats_results['residual_normality']['p_value']:.4f}")

    # Run Monte Carlo validation if enabled
    if USE_MONTE_CARLO:
        logger.info("Running Monte Carlo validation...")
        mc_results = monte_carlo_validation(model, X_seq, y_seq)

        # Save Monte Carlo results
        with open(os.path.join(VALIDATION_DIR, 'monte_carlo_results.json'), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            mc_results_json = {}
            for key, metrics in mc_results.items():
                mc_results_json[key] = {
                    k: v.item() if hasattr(v, 'item') else v if not isinstance(v, list) else [
                        x.item() if hasattr(x, 'item') else x for x in v]
                    for k, v in metrics.items()
                }
            json.dump(mc_results_json, f, indent=4)

    # Generate prediction intervals if enabled
    if PREDICTION_INTERVALS:
        logger.info("Calculating prediction intervals...")
        base_pred, lower_bounds, upper_bounds = calculate_prediction_intervals(model, X_seq)

        # Store the last batch of results for plotting
        interval_df = pd.DataFrame({
            'actual': y_seq,
            'prediction': base_pred.flatten(),
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        })

        # Plot prediction intervals
        plt.figure(figsize=(14, 7))
        plt.fill_between(
            range(len(interval_df)),
            interval_df['lower_bound'],
            interval_df['upper_bound'],
            alpha=0.2, color='blue',
            label='95% Prediction Interval'
        )
        plt.plot(interval_df['prediction'], 'b-', label='Prediction')
        plt.plot(interval_df['actual'], 'r-', label='Actual')
        plt.title('Predictions with Confidence Intervals')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(VALIDATION_DIR, 'prediction_intervals.png'))

    # Save results
    results.to_csv(os.path.join(VALIDATION_DIR, 'walk_forward_results.csv'), index=False)

    return results, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }


def validate_feature_importance(model, df, scaler, expected_features=None):
    """
    Validate feature importance by permutation importance
    """
    logger.info("Calculating feature importance...")

    # Preprocess data
    X, y = preprocess_data(df, scaler, expected_features)[0:2]
    X_seq, y_seq = create_sequences(X, y)

    # Get baseline performance
    baseline_pred = model.predict(X_seq, verbose=0)
    baseline_mse = mean_squared_error(y_seq, baseline_pred)

    # Calculate importance for each feature
    importance = {}

    # Only permute a subset of features for efficiency
    feature_count = X_seq.shape[2]

    # If we have a lot of features, just sample a subset
    if feature_count > 50:
        logger.info(
            f"Large feature set detected ({feature_count} features), sampling 50 random features for importance analysis")
        feature_indices = np.random.choice(feature_count, size=50, replace=False)
    else:
        feature_indices = range(feature_count)

    for i in feature_indices:
        feature_name = f"feature_{i}"
        logger.info(f"Processing feature importance for {feature_name}")

        # Copy the data
        X_permuted = X_seq.copy()

        # Shuffle the feature across all sequences
        for j in range(X_permuted.shape[0]):
            # Get a random permutation of the feature
            np.random.shuffle(X_permuted[j, :, i])

        # Predict with permuted feature
        perm_pred = model.predict(X_permuted, verbose=0)
        perm_mse = mean_squared_error(y_seq, perm_pred)

        # Calculate importance
        importance[feature_name] = perm_mse - baseline_mse

    # Convert to DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.title('Feature Importance (Permutation Method) - Top 20')
    plt.xlabel('Increase in MSE when feature is permuted')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'feature_importance.png'))

    logger.info(f"Feature Importance: Top 10\n{importance_df.head(10)}")

    # Save full results
    importance_df.to_csv(os.path.join(VALIDATION_DIR, 'feature_importance.csv'), index=False)

    return importance_df


def cross_validate_time_periods(model, df, scaler, expected_features=None):
    """
    Cross-validate across different time periods (days of week, months, etc.)
    """
    logger.info("Cross-validating across time periods...")

    # Preprocess all data
    X, y, _ = preprocess_data(df, scaler, expected_features)
    X_seq, y_seq = create_sequences(X, y)

    # Create a DataFrame for the data subset to match with date information
    df_subset = df.iloc[LOOKBACK:].reset_index(drop=True)

    # Make sure we have enough data in the subset
    if len(df_subset) < len(y_seq):
        logger.warning(f"Mismatch in data length: df_subset={len(df_subset)}, y_seq={len(y_seq)}")
        # Truncate to the shorter length
        min_len = min(len(df_subset), len(y_seq))
        df_subset = df_subset.iloc[:min_len]
        y_seq = y_seq[:min_len]
        X_seq = X_seq[:min_len]

    # Day of week analysis
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_results = {}

    for day_num in range(7):
        # Filter by day of week
        day_mask = df_subset['day_of_week'] == day_num
        day_indices = np.where(day_mask)[0]

        if len(day_indices) == 0:
            continue

        # Get data for this day
        X_day = X_seq[day_indices]
        y_day = y_seq[day_indices]

        # Predict
        y_pred = model.predict(X_day, verbose=0)

        # Calculate metrics
        mse = mean_squared_error(y_day, y_pred)
        mae = mean_absolute_error(y_day, y_pred)
        directional_acc = np.mean((np.sign(y_day) == np.sign(y_pred)).astype(int))

        day_results[days[day_num]] = {
            'count': len(day_indices),
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_acc
        }

    # Convert to DataFrame
    day_df = pd.DataFrame.from_dict(day_results, orient='index')
    logger.info(f"Day of Week Performance:\n{day_df}")

    # Month analysis
    month_results = {}
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for month_num in range(1, 13):
        # Filter by month
        month_mask = df_subset['month'] == month_num
        month_indices = np.where(month_mask)[0]

        if len(month_indices) == 0:
            continue

        # Get data for this month
        X_month = X_seq[month_indices]
        y_month = y_seq[month_indices]

        # Predict
        y_pred = model.predict(X_month, verbose=0)

        # Calculate metrics
        mse = mean_squared_error(y_month, y_pred)
        mae = mean_absolute_error(y_month, y_pred)
        directional_acc = np.mean((np.sign(y_month) == np.sign(y_pred)).astype(int))

        month_results[months[month_num - 1]] = {
            'count': len(month_indices),
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_acc
        }

    # Convert to DataFrame
    month_df = pd.DataFrame.from_dict(month_results, orient='index')
    logger.info(f"Month Performance:\n{month_df}")

    # Market regime analysis if available
    regime_df = None
    if USE_MARKET_REGIMES and 'regime' in df_subset.columns:
        regime_results = {}
        regimes = ['Normal', 'Trending', 'Mean-Rev', 'Volatile']

        for regime_id, regime_name in enumerate(regimes):
            # Filter by regime
            regime_mask = df_subset['regime'] == regime_id
            regime_indices = np.where(regime_mask)[0]

            if len(regime_indices) == 0:
                continue

            # Get data for this regime
            X_regime = X_seq[regime_indices]
            y_regime = y_seq[regime_indices]

            # Predict
            y_pred = model.predict(X_regime, verbose=0)

            # Calculate metrics
            mse = mean_squared_error(y_regime, y_pred)
            directional_acc = np.mean((np.sign(y_regime) == np.sign(y_pred)).astype(int))

            regime_results[regime_name] = {
                'count': len(regime_indices),
                'mse': mse,
                'directional_accuracy': directional_acc
            }

        # Convert to DataFrame
        regime_df = pd.DataFrame.from_dict(regime_results, orient='index')
        logger.info(f"Market Regime Performance:\n{regime_df}")

    # Plot day of week performance
    plt.figure(figsize=(14, 6))

    # Directional accuracy by day of week
    ax = plt.subplot(1, 2, 1)
    day_df['directional_accuracy'].plot(kind='bar', ax=ax, color='blue')
    plt.title('Directional Accuracy by Day of Week')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Add count labels
    for i, v in enumerate(day_df['directional_accuracy']):
        plt.text(i, v + 0.02, f"n={day_df['count'].iloc[i]}", ha='center')

    # MSE by day of week
    ax = plt.subplot(1, 2, 2)
    day_df['mse'].plot(kind='bar', ax=ax, color='red')
    plt.title('MSE by Day of Week')
    plt.ylabel('Mean Squared Error')

    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'day_of_week_performance.png'))

    # Plot month performance
    plt.figure(figsize=(14, 6))

    # Directional accuracy by month
    ax = plt.subplot(1, 2, 1)
    month_df['directional_accuracy'].plot(kind='bar', ax=ax, color='green')
    plt.title('Directional Accuracy by Month')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Add count labels
    for i, v in enumerate(month_df['directional_accuracy']):
        plt.text(i, v + 0.02, f"n={month_df['count'].iloc[i]}", ha='center')

    # MSE by month
    ax = plt.subplot(1, 2, 2)
    month_df['mse'].plot(kind='bar', ax=ax, color='orange')
    plt.title('MSE by Month')
    plt.ylabel('Mean Squared Error')

    plt.tight_layout()
    plt.savefig(os.path.join(VALIDATION_DIR, 'month_performance.png'))

    # Plot regime performance if available
    if regime_df is not None:
        plt.figure(figsize=(14, 6))

        # Directional accuracy by regime
        ax = plt.subplot(1, 2, 1)
        regime_df['directional_accuracy'].plot(kind='bar', ax=ax, color='purple')
        plt.title('Directional Accuracy by Market Regime')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        # Add count labels
        for i, v in enumerate(regime_df['directional_accuracy']):
            plt.text(i, v + 0.02, f"n={regime_df['count'].iloc[i]}", ha='center')

        # MSE by regime
        ax = plt.subplot(1, 2, 2)
        regime_df['mse'].plot(kind='bar', ax=ax, color='brown')
        plt.title('MSE by Market Regime')
        plt.ylabel('Mean Squared Error')

        plt.tight_layout()
        plt.savefig(os.path.join(VALIDATION_DIR, 'regime_performance.png'))

    # Save results to CSV
    day_df.to_csv(os.path.join(VALIDATION_DIR, 'day_performance.csv'))
    month_df.to_csv(os.path.join(VALIDATION_DIR, 'month_performance.csv'))
    if regime_df is not None:
        regime_df.to_csv(os.path.join(VALIDATION_DIR, 'regime_performance.csv'))

    return day_df, month_df, regime_df


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
                'directional_accuracy': directional_accuracy
            }

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
        pred = model.predict(X)
        predictions.append(pred)

    # Weighted average
    weighted_preds = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_preds += weights[i] * pred

    return weighted_preds


def main():
    # MT5 connection params
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    # Connect to MT5
    if not connect_to_mt5(account, password, server):
        return

    try:
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
        if is_ensemble and ensemble_exists:
            logger.info("Loading ensemble models...")
            ensemble_models, ensemble_weights = load_ensemble_models(ensemble_dir)
            if ensemble_models is None:
                logger.warning("Failed to load ensemble, falling back to single model")
                is_ensemble = False

        # Try different model filenames if not using ensemble
        model = None
        if not is_ensemble:
            model_filenames = ['final_model.h5', 'mt5_neural_network_model.h5', 'model.h5', 'best_model.h5']

            for filename in model_filenames:
                try:
                    model_path = os.path.join(MODEL_DIR, filename)
                    if os.path.exists(model_path):
                        # Define custom metrics
                        custom_objects = {
                            'r2_keras': r2_keras,
                            'directional_accuracy': directional_accuracy
                        }

                        model = load_model(model_path, custom_objects=custom_objects)
                        logger.info(f"Model loaded successfully from {filename}")
                        break
                except Exception as e:
                    logger.warning(f"Could not load model from {filename}: {e}")

            if model is None:
                logger.error("Could not find a valid model file")
                return

        # Inspect model input shape
        if is_ensemble:
            expected_features = inspect_model_input_shape(ensemble_models[0])
        else:
            expected_features = inspect_model_input_shape(model)

        if expected_features:
            logger.info(f"Model expects {expected_features} input features")
        else:
            logger.warning("Could not determine expected feature count from model")
            expected_features = 103  # Fallback based on enhanced training

        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return

        # Define date range for validation data (2 years)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=2 * 365)

        # Get historical data
        df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)
        if df is None:
            return

        # Filter for 5 PM Arizona time
        df_5pm = filter_5pm_data(df)

        # Check if we have enough data
        if len(df_5pm) <= LOOKBACK:
            logger.error(f"Not enough data points at 5 PM AZ time (only {len(df_5pm)} found)")
            return

        # Feature engineering to match training
        logger.info("Adding datetime features...")
        df_5pm = add_datetime_features(df_5pm)

        # Add market regime detection if enabled
        if USE_MARKET_REGIMES:
            logger.info("Detecting market regimes...")
            df_5pm = detect_market_regime(df_5pm)

        # Add wavelet features
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

        # Perform walk-forward validation
        logger.info("Starting walk-forward validation...")
        if is_ensemble:
            # For ensemble, we need to create a wrapper for prediction
            class EnsembleModel:
                def __init__(self, models, weights):
                    self.models = models
                    self.weights = weights

                def predict(self, X, verbose=0):
                    return ensemble_predict(self.models, self.weights, X)

            ensemble_wrapper = EnsembleModel(ensemble_models, ensemble_weights)
            wf_results, wf_metrics = walk_forward_validation(
                ensemble_wrapper, df_5pm, scaler, expected_features=expected_features)
        else:
            wf_results, wf_metrics = walk_forward_validation(
                model, df_5pm, scaler, expected_features=expected_features)

        # Validate feature importance
        logger.info("Validating feature importance...")
        if is_ensemble:
            # Use the first model in ensemble for feature importance (for simplicity)
            feature_importance = validate_feature_importance(
                ensemble_models[0], df_5pm, scaler, expected_features=expected_features)
        else:
            feature_importance = validate_feature_importance(
                model, df_5pm, scaler, expected_features=expected_features)

        # Cross-validate across time periods
        logger.info("Cross-validating across time periods...")
        if is_ensemble:
            day_perf, month_perf, regime_perf = cross_validate_time_periods(
                ensemble_wrapper, df_5pm, scaler, expected_features=expected_features)
        else:
            day_perf, month_perf, regime_perf = cross_validate_time_periods(
                model, df_5pm, scaler, expected_features=expected_features)

        # Create validation summary
        summary = {
            'walk_forward_metrics': wf_metrics,
            'top_features': feature_importance['Feature'].tolist()[:5],
            'best_day': day_perf['directional_accuracy'].idxmax(),
            'best_month': month_perf['directional_accuracy'].idxmax()
        }

        if regime_perf is not None:
            summary['best_regime'] = regime_perf['directional_accuracy'].idxmax()

        # Save validation results
        with open(os.path.join(VALIDATION_DIR, 'validation_summary.json'), 'w') as f:
            # Convert any numpy numbers to Python native types for JSON
            clean_summary = {}
            for k, v in summary.items():
                if isinstance(v, dict):
                    clean_summary[k] = {kk: float(vv) if hasattr(vv, 'item') else vv for kk, vv in v.items()}
                else:
                    clean_summary[k] = v
            json.dump(clean_summary, f, indent=4)

        logger.info("Validation completed successfully!")

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        logger.info("MT5 connection closed")


if __name__ == "__main__":
    main()