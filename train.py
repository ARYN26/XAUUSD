#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Training Script with Attention Mechanisms
Focus: 5 PM Arizona Time Data with Advanced Feature Engineering and Ensemble Techniques
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
from sklearn.ensemble import RandomForestRegressor
import MetaTrader5 as mt5
import pytz
import pickle
import logging
from scipy import stats
import warnings
import tensorflow.keras.backend as K
import json

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
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Model type options
MODEL_TYPES = ['lstm', 'gru', 'bidirectional', 'cnn_lstm', 'attention']

# Additional Parameters
ENSEMBLE_SIZE = 3  # Number of models in ensemble
USE_WAVELET = True  # Whether to use wavelet transformation for feature extraction
MARKET_REGIMES = True  # Whether to detect market regimes


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
    Add cyclical datetime features
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
    Based on research paper findings on regime-based trading
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=window).std()

    # Calculate autocorrelation - negative values suggest mean reversion
    df['autocorrelation'] = df['returns'].rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    # Calculate trend strength using Hurst exponent approximation
    def hurst_exponent(returns, lags=range(2, 20)):
        tau = []
        std = []
        for lag in lags:
            # Construct a new series with lagged returns
            series_lagged = pd.Series(returns).diff(lag).dropna()
            tau.append(lag)
            std.append(np.std(series_lagged))

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
        for window in [5, 10, 20, 50, 100, 200]:  # Added 200-day MA
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:  # Added 200-day EMA
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
        for window in [5, 10, 20, 50, 200]:  # Added 200-day ratio
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

    return df


def prepare_features_and_targets(df, target_col='next_close_change_pct', feature_blacklist=None):
    """
    Prepare features and target variables with more robust handling of blacklisted features
    """
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

    # Separate features and target
    y = feature_df[target_col].values

    # Remove target columns from features
    target_cols = ['next_close_change_pct', 'next_direction', 'future_volatility', 'extreme_move_5d', 'regime_switch',
                   'next_regime']
    target_cols += [f'change_future_{i}_pct' for i in range(2, 6)]
    X = feature_df.drop(columns=target_cols, errors='ignore').values

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Feature names: {feature_df.drop(columns=target_cols, errors='ignore').columns.tolist()}")

    return X, y, feature_df.drop(columns=target_cols, errors='ignore').columns.tolist()


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


def build_attention_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build LSTM model with attention mechanism based on research findings
    """
    # Fix: Ensure all complexity levels have at least 3 elements in units list
    if complexity == 'low':
        units = [64, 48, 32]  # Added a middle value to avoid index error
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

    # Third LSTM layer - now units[2] will always exist
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

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', directional_accuracy, r2_keras]
    )

    return model


def build_cnn_lstm_model(input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Build hybrid CNN-LSTM model based on research findings
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

    # Compile
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

    # Compile
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

    # Compile
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

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', directional_accuracy, r2_keras]
    )

    return model


def build_traditional_model(X_train, y_train):
    """
    Build a traditional ML model (Random Forest) for comparison/ensemble
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    model.fit(X_train, y_train)

    return model


def build_model_by_type(model_type, input_shape, complexity='medium', dropout_rate=0.3, learning_rate=0.001):
    """
    Factory function to build models based on type
    """
    if model_type == 'lstm':
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


def hyperparameter_grid_search(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape):
    """
    Enhanced hyperparameter grid search with new model types
    """
    # Define hyperparameter grid
    hyperparams = {
        'model_type': MODEL_TYPES,  # Now includes new model types
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'complexity': ['low', 'medium', 'high']
    }

    best_val_loss = float('inf')
    best_params = {}
    best_model = None

    # Log total combinations
    total_combinations = (
            len(hyperparams['model_type']) *
            len(hyperparams['dropout_rate']) *
            len(hyperparams['learning_rate']) *
            len(hyperparams['complexity'])
    )
    logger.info(f"Starting grid search with {total_combinations} combinations")

    # Try all combinations
    for model_type in hyperparams['model_type']:
        for dropout_rate in hyperparams['dropout_rate']:
            for learning_rate in hyperparams['learning_rate']:
                for complexity in hyperparams['complexity']:
                    # Create model
                    K.clear_session()

                    current_params = {
                        'model_type': model_type,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'complexity': complexity
                    }

                    logger.info(f"Training with parameters: {current_params}")

                    # Build model based on type
                    model = build_model_by_type(
                        model_type=model_type,
                        input_shape=input_shape,
                        complexity=complexity,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )

                    # Train with early stopping
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )

                    history = model.fit(
                        X_train_seq, y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # Check if this model is better
                    val_loss = history.history['val_loss'][-1]
                    val_dir_acc = history.history['val_directional_accuracy'][-1]

                    logger.info(f"Val Loss: {val_loss:.6f}, Val Dir Acc: {val_dir_acc:.2%}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = current_params
                        best_model = model
                        logger.info(f"New best model found!")

                    # Force garbage collection
                    import gc
                    gc.collect()

    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return best_model, best_params


def build_ensemble_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape, best_params, ensemble_size=3):
    """
    Build an ensemble of models based on research findings of improved ensemble accuracy
    """
    models = []
    weights = []

    # Base model with best params
    model_type = best_params['model_type']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    complexity = best_params['complexity']

    logger.info(f"Building ensemble with {ensemble_size} models")

    # Create first model with best params
    first_model = build_model_by_type(
        model_type=model_type,
        input_shape=input_shape,
        complexity=complexity,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # Train first model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    first_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    models.append(first_model)

    # Initial validation performance
    val_pred = first_model.predict(X_val_seq)
    val_loss = mean_squared_error(y_val_seq, val_pred)
    weights.append(1.0 / val_loss)

    # Add diversity with different model types and parameters
    for i in range(1, ensemble_size):
        # Create a different model variant
        if i % 2 == 0:
            # Different model type
            alt_model_type = np.random.choice([t for t in MODEL_TYPES if t != model_type])
            alt_model = build_model_by_type(
                model_type=alt_model_type,
                input_shape=input_shape,
                complexity=complexity,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
        else:
            # Different hyperparameters
            alt_dropout = dropout_rate + np.random.choice([-0.1, 0.1])
            alt_dropout = max(0.1, min(0.5, alt_dropout))  # Keep in reasonable range

            alt_lr = learning_rate * np.random.choice([0.5, 2.0])
            alt_lr = max(0.00005, min(0.002, alt_lr))  # Keep in reasonable range

            alt_model = build_model_by_type(
                model_type=model_type,
                input_shape=input_shape,
                complexity=complexity,
                dropout_rate=alt_dropout,
                learning_rate=alt_lr
            )

        # Train model
        alt_model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        models.append(alt_model)

        # Add weight based on validation performance
        val_pred = alt_model.predict(X_val_seq)
        val_loss = mean_squared_error(y_val_seq, val_pred)
        weights.append(1.0 / val_loss)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    logger.info(f"Ensemble model weights: {normalized_weights}")

    return models, normalized_weights


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


def train_final_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, batch_size=32, epochs=100):
    """
    Train the final model with all callbacks
    """
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
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test_seq, y_test_seq, is_ensemble=False, ensemble_models=None, ensemble_weights=None):
    """
    Evaluate the model on test data
    """
    # Predictions
    if is_ensemble and ensemble_models is not None and ensemble_weights is not None:
        y_pred = ensemble_predict(ensemble_models, ensemble_weights, X_test_seq)
    else:
        y_pred = model.predict(X_test_seq)

    # Calculate metrics
    mse = mean_squared_error(y_test_seq, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)

    # Directional accuracy
    directional_acc = np.mean((np.sign(y_test_seq) == np.sign(y_pred)).astype(int))

    logger.info(f"Test Metrics:")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    logger.info(f"Directional Accuracy: {directional_acc:.2%}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_acc,
        'predictions': y_pred,
        'actual': y_test_seq
    }


def calculate_feature_importance(model, X_test_seq, y_test_seq, feature_list):
    """
    Calculate feature importance using permutation method
    """
    # Baseline performance
    y_pred = model.predict(X_test_seq)
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
        y_pred_permuted = model.predict(X_permuted)
        permuted_mse = mean_squared_error(y_test_seq, y_pred_permuted)

        # Importance is the increase in error
        importance = permuted_mse - baseline_mse
        importances.append(importance)

    return importances


def plot_results(history, test_results, feature_list, market_regimes=False):
    """
    Plot training history and test results with enhanced visualizations
    """
    # Training history
    plt.figure(figsize=(20, 15))

    # Plot loss
    plt.subplot(3, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot MAE
    plt.subplot(3, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

    # Plot directional accuracy
    plt.subplot(3, 3, 3)
    plt.plot(history.history['directional_accuracy'], label='Training Dir. Accuracy')
    plt.plot(history.history['val_directional_accuracy'], label='Validation Dir. Accuracy')
    plt.title('Directional Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot predictions vs actual
    plt.subplot(3, 3, 4)
    plt.plot(test_results['actual'], label='Actual', alpha=0.7)
    plt.plot(test_results['predictions'], label='Predictions', alpha=0.7)
    plt.title('Test Set: Predictions vs Actual')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Value')

    # Plot prediction error
    plt.subplot(3, 3, 5)
    error = test_results['actual'] - test_results['predictions'].flatten()
    plt.plot(error)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Prediction Error')
    plt.xlabel('Sample')
    plt.ylabel('Error')

    # Plot scatter plot
    plt.subplot(3, 3, 6)
    plt.scatter(test_results['actual'], test_results['predictions'])
    plt.axline([0, 0], [1, 1], color='r', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Plot error distribution
    plt.subplot(3, 3, 7)
    plt.hist(error, bins=30, alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')

    # Plot cumulative returns
    plt.subplot(3, 3, 8)

    # Model strategy returns (go long/short based on predicted direction)
    strategy_returns = np.sign(test_results['predictions'].flatten()) * test_results['actual']

    # Buy and hold returns
    buy_hold_returns = test_results['actual']

    # Cumulative returns
    cum_strategy = np.cumsum(strategy_returns)
    cum_buy_hold = np.cumsum(buy_hold_returns)

    plt.plot(cum_strategy, label='Model Strategy', color='green')
    plt.plot(cum_buy_hold, label='Buy & Hold', color='blue', alpha=0.7)
    plt.title('Cumulative Returns')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Cumulative Return %')

    # Plot trade analysis
    plt.subplot(3, 3, 9)

    # Calculate metrics
    total_trades = len(strategy_returns)
    winning_trades = np.sum(strategy_returns > 0)
    losing_trades = np.sum(strategy_returns < 0)
    win_ratio = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = np.mean(strategy_returns[strategy_returns > 0]) if any(strategy_returns > 0) else 0
    avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if any(strategy_returns < 0) else 0

    # Create a trade metrics table
    from matplotlib.table import Table
    ax = plt.gca()
    ax.axis('off')

    table_data = [
        ['Total Trades', f"{total_trades}"],
        ['Winning Trades', f"{winning_trades} ({win_ratio:.2%})"],
        ['Losing Trades', f"{losing_trades} ({1 - win_ratio:.2%})"],
        ['Avg Win', f"{avg_win:.2f}%"],
        ['Avg Loss', f"{avg_loss:.2f}%"],
        ['Final Return', f"{cum_strategy[-1]:.2f}%"],
        ['Buy & Hold', f"{cum_buy_hold[-1]:.2f}%"]
    ]

    table = Table(ax, bbox=[0, 0, 1, 1])

    for i, (name, value) in enumerate(table_data):
        table.add_cell(i, 0, 0.7, 0.1, text=name, loc='right')
        table.add_cell(i, 1, 0.3, 0.1, text=value, loc='right')

    ax.add_table(table)
    plt.title('Trading Performance')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'))

    # Plot feature importance if available
    if 'feature_importance' in test_results:
        # Sort by importance
        importance_df = pd.DataFrame({
            'Feature': feature_list,
            'Importance': test_results['feature_importance']
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
        plt.title('Feature Importance (Top 20)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))

    # Plot market regimes if available
    if market_regimes and 'regime_data' in test_results:
        regime_data = test_results['regime_data']

        plt.figure(figsize=(15, 10))

        # Plot regime distribution
        plt.subplot(2, 2, 1)
        regime_counts = regime_data['regime'].value_counts().sort_index()
        regimes = ['Normal', 'Trending', 'Mean-Rev', 'Volatile']
        plt.bar(regimes, regime_counts)
        plt.title('Market Regime Distribution')
        plt.xlabel('Regime Type')
        plt.ylabel('Count')

        # Plot performance by regime
        plt.subplot(2, 2, 2)
        regime_perf = []

        for i, regime in enumerate(['Normal', 'Trending', 'Mean-Rev', 'Volatile']):
            regime_idx = regime_data['regime'] == i
            if any(regime_idx):
                actual = test_results['actual'][regime_idx]
                pred = test_results['predictions'].flatten()[regime_idx]
                dir_acc = np.mean((np.sign(actual) == np.sign(pred)).astype(int))
                regime_perf.append(dir_acc)
            else:
                regime_perf.append(0)

        plt.bar(regimes, regime_perf)
        plt.title('Directional Accuracy by Regime')
        plt.xlabel('Regime Type')
        plt.ylabel('Accuracy')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)

        # Plot regime transitions
        plt.subplot(2, 2, 3)
        regime_data['next_regime'] = regime_data['regime'].shift(-1)
        transition_counts = regime_data.groupby(['regime', 'next_regime']).size().unstack(fill_value=0)

        plt.matshow(transition_counts, fignum=False, cmap='viridis')
        plt.colorbar(label='Count')
        plt.title('Regime Transition Matrix')
        plt.xlabel('Next Regime')
        plt.ylabel('Current Regime')

        # Plot regime over time
        plt.subplot(2, 2, 4)
        for i, regime in enumerate(['Normal', 'Trending', 'Mean-Rev', 'Volatile']):
            regime_idx = regime_data['regime'] == i
            plt.scatter(np.where(regime_idx)[0], [i] * sum(regime_idx), label=regime, alpha=0.7)

        plt.title('Regime Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Regime')
        plt.yticks(range(4), regimes)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'market_regimes.png'))


def save_model_and_metadata(model, scaler, feature_list, best_params, test_metrics, is_ensemble=False,
                            ensemble_models=None, ensemble_weights=None):
    """
    Save model, scaler, feature list, and metadata with support for ensemble models
    """
    # Save model architecture as JSON for better portability
    if is_ensemble and ensemble_models is not None:
        # Create ensemble directory
        ensemble_dir = os.path.join(MODEL_DIR, 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)

        # Save each model in the ensemble
        for i, model in enumerate(ensemble_models):
            model.save(os.path.join(ensemble_dir, f'model_{i}.h5'))

        # Save ensemble weights
        with open(os.path.join(ensemble_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(ensemble_weights, f)
    else:
        # Save single model
        model.save(os.path.join(MODEL_DIR, 'final_model.h5'))

        # Save model architecture separately
        model_json = model.to_json()
        with open(os.path.join(MODEL_DIR, 'model_architecture.json'), 'w') as f:
            f.write(model_json)

    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature list
    with open(os.path.join(MODEL_DIR, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(feature_list, f)

    # Convert test_metrics to JSON-serializable format
    serializable_metrics = {}
    for k, v in test_metrics.items():
        if k not in ['predictions', 'actual', 'regime_data']:
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
        'feature_count': int(len(feature_list)),  # Ensure it's a regular int
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_ensemble': bool(is_ensemble),  # Convert to regular Python bool
        'ensemble_size': int(len(ensemble_models) if is_ensemble and ensemble_models else 1),
        'model_type': serializable_params.get('model_type', 'ensemble') if not is_ensemble else 'ensemble',
        'symbol': SYMBOL,
        'lookback': int(LOOKBACK)  # Ensure it's a regular int
    }

    # Save metadata
    try:
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model and metadata saved to {MODEL_DIR}")
    except TypeError as e:
        logger.error(f"Error saving metadata: {e}")
        # Print the metadata content for debugging
        for key, value in metadata.items():
            logger.info(f"Metadata key '{key}' has type {type(value)}")
            if isinstance(value, dict):
                for k, v in value.items():
                    logger.info(f"  Subkey '{k}' has type {type(v)}")


def main():
    # MT5 connection params
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    # Connect to MT5
    if not connect_to_mt5(account, password, server):
        return

    try:
        # Define date range for historical data (4 years - increased for more training data)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=4 * 365)

        # Get historical data
        df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)
        if df is None:
            return

        # Filter for 5 PM Arizona time
        df_5pm = filter_5pm_data(df)

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

        # Split data
        train_df, val_df, test_df = time_series_split(df_5pm)

        # Prepare features and targets
        X_train, y_train, feature_list = prepare_features_and_targets(train_df)
        X_val, y_val, _ = prepare_features_and_targets(val_df)
        X_test, y_test, _ = prepare_features_and_targets(test_df)

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test, scaler_type='robust'
        )

        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, LOOKBACK)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, LOOKBACK)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, LOOKBACK)

        logger.info(f"Sequence shapes: Train {X_train_seq.shape}, Val {X_val_seq.shape}, Test {X_test_seq.shape}")

        # Hyperparameter search
        logger.info("Starting hyperparameter grid search...")
        best_model, best_params = hyperparameter_grid_search(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])
        )

        # Build ensemble model
        logger.info("Building ensemble model...")
        ensemble_models, ensemble_weights = build_ensemble_model(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
            best_params=best_params,
            ensemble_size=ENSEMBLE_SIZE
        )

        # Train final (best individual) model
        logger.info("Training final individual model with optimal hyperparameters...")
        final_model, history = train_final_model(
            best_model,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            batch_size=32,
            epochs=100
        )

        # Evaluate individual model on test data
        logger.info("Evaluating final individual model on test data...")
        ind_test_results = evaluate_model(final_model, X_test_seq, y_test_seq)

        # Evaluate ensemble model on test data
        logger.info("Evaluating ensemble model on test data...")
        ens_test_results = evaluate_model(
            None, X_test_seq, y_test_seq,
            is_ensemble=True, ensemble_models=ensemble_models, ensemble_weights=ensemble_weights
        )

        # Compare individual vs ensemble performance
        logger.info(f"Individual vs Ensemble performance:")
        logger.info(
            f"Individual MSE: {ind_test_results['mse']:.6f}, Dir Acc: {ind_test_results['directional_accuracy']:.2%}")
        logger.info(
            f"Ensemble MSE: {ens_test_results['mse']:.6f}, Dir Acc: {ens_test_results['directional_accuracy']:.2%}")

        # Choose the best model based on directional accuracy
        use_ensemble = ens_test_results['directional_accuracy'] > ind_test_results['directional_accuracy']
        best_results = ens_test_results if use_ensemble else ind_test_results
        logger.info(f"Using {'ensemble' if use_ensemble else 'individual'} model as it performs better")

        # Calculate feature importance for final model
        logger.info("Calculating feature importance...")
        if use_ensemble:
            # For ensemble, calculate feature importance using the best performing model
            best_ensemble_idx = np.argmax([1.0 / w for w in ensemble_weights])
            feature_importance = calculate_feature_importance(
                ensemble_models[best_ensemble_idx], X_test_seq, y_test_seq, feature_list
            )
        else:
            feature_importance = calculate_feature_importance(
                final_model, X_test_seq, y_test_seq, feature_list
            )

        best_results['feature_importance'] = feature_importance

        # Add market regime data if enabled
        if MARKET_REGIMES:
            test_with_regimes = test_df.iloc[LOOKBACK:].reset_index(drop=True)
            regime_cols = ['regime', 'regime_trending', 'regime_mean_reverting', 'regime_volatile']
            regime_data = test_with_regimes[regime_cols]
            best_results['regime_data'] = regime_data

        # Plot results
        logger.info("Plotting results...")
        if use_ensemble:
            # Create a mock history object for ensemble (since we don't have a single history)
            class EnsembleHistory:
                def __init__(self, individual_history):
                    self.history = {
                        'loss': individual_history.history['loss'],
                        'val_loss': individual_history.history['val_loss'],
                        'mae': individual_history.history['mae'],
                        'val_mae': individual_history.history['val_mae'],
                        'directional_accuracy': individual_history.history['directional_accuracy'],
                        'val_directional_accuracy': individual_history.history['val_directional_accuracy']
                    }

            ensemble_history = EnsembleHistory(history)
            plot_results(ensemble_history, best_results, feature_list, market_regimes=MARKET_REGIMES)
        else:
            plot_results(history, best_results, feature_list, market_regimes=MARKET_REGIMES)

        # Save model and metadata
        logger.info("Saving model and metadata...")
        save_model_and_metadata(
            final_model, scaler, feature_list, best_params, best_results,
            is_ensemble=use_ensemble, ensemble_models=ensemble_models, ensemble_weights=ensemble_weights
        )

        # Create trading strategy configuration
        strategy_config = {
            'model_type': 'ensemble' if use_ensemble else best_params['model_type'],
            'lookback': LOOKBACK,
            'features': feature_list,
            'symbol': SYMBOL,
            'timeframe': 'H1',  # 1-hour timeframe
            'target_hour': TARGET_HOUR,
            'timezone': 'US/Arizona',
            'top_features': [feature_list[i] for i in np.argsort(feature_importance)[-10:]],
            'entry_threshold': 0.1,  # Only take trades with predicted change > 0.1%
            'use_market_regimes': MARKET_REGIMES,
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save strategy configuration
        with open(os.path.join(MODEL_DIR, 'strategy_config.json'), 'w') as f:
            json.dump(strategy_config, f, indent=4)

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