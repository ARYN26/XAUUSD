#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced MT5 Neural Network Training Script
Focus: 5 PM Arizona Time Data with Advanced Feature Engineering
No TA-Lib dependency
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MetaTrader5 as mt5
import pytz
import pickle
import logging
from scipy import stats
import warnings
import tensorflow.keras.backend as K

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
LOOKBACK = 20  # Number of previous time periods to consider
SYMBOL = 'EURUSD'  # Trading symbol
TIMEFRAME = mt5.TIMEFRAME_H1  # 1-hour timeframe

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time

# Define paths
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


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
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        # Price relative to moving averages
        for window in [5, 10, 20, 50]:
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']

        # Moving average crossovers
        df['sma_5_10_cross'] = np.where(df['sma_5'] > df['sma_10'], 1, -1)
        df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['ema_5_10_cross'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)

        # Volatility indicators
        df['volatility_10'] = df['close_diff_pct'].rolling(window=10).std()
        df['volatility_20'] = df['close_diff_pct'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and loss
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()

            # Calculate relative strength
            rs = avg_gain / avg_loss

            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))

            return rsi

        df['rsi_14'] = calculate_rsi(df['close'], window=14)

        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            # Calculate middle band (simple moving average)
            middle_band = prices.rolling(window=window).mean()

            # Calculate standard deviation
            std = prices.rolling(window=window).std()

            # Calculate upper and lower bands
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)

            # Calculate Bollinger Band width
            bb_width = (upper_band - lower_band) / middle_band

            # Calculate BB position
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
            # Calculate fast and slow EMA
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        macd_line, signal_line, histogram = calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Stochastic Oscillator
        def calculate_stochastic(high, low, close, k_period=14, d_period=3):
            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()

            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

            # Calculate %D (simple moving average of %K)
            d = k.rolling(window=d_period).mean()

            return k, d

        k, d = calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k
        df['stoch_d'] = d

        # Average True Range (ATR)
        def calculate_atr(high, low, close, window=14):
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = tr.rolling(window=window).mean()

            return atr

        df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'])

        # Money Flow Index (MFI)
        def calculate_mfi(high, low, close, volume, window=14):
            # Calculate typical price
            tp = (high + low + close) / 3

            # Calculate raw money flow
            raw_money_flow = tp * volume

            # Get the direction of money flow
            money_flow_positive = np.where(tp > tp.shift(1), raw_money_flow, 0)
            money_flow_negative = np.where(tp < tp.shift(1), raw_money_flow, 0)

            # Convert to series
            money_flow_positive = pd.Series(money_flow_positive, index=high.index)
            money_flow_negative = pd.Series(money_flow_negative, index=high.index)

            # Calculate money flow ratio
            positive_sum = money_flow_positive.rolling(window=window).sum()
            negative_sum = money_flow_negative.rolling(window=window).sum()

            money_flow_ratio = positive_sum / negative_sum

            # Calculate MFI
            mfi = 100 - (100 / (1 + money_flow_ratio))

            return mfi

        df['mfi_14'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])

        # Commodity Channel Index (CCI)
        def calculate_cci(high, low, close, window=20):
            # Calculate typical price
            tp = (high + low + close) / 3

            # Calculate simple moving average of typical price
            sma_tp = tp.rolling(window=window).mean()

            # Calculate mean deviation
            mean_deviation = abs(tp - sma_tp).rolling(window=window).mean()

            # Calculate CCI
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

        # Volume indicators
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']

        # Z-score of price
        df['zscore_20'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return df


def add_lagged_features(df, lags=[1, 2, 3, 5]):
    """
    Add lagged features for selected columns
    """
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'close_diff_lag_{lag}'] = df['close_diff'].shift(lag)
        df[f'close_diff_pct_lag_{lag}'] = df['close_diff_pct'].shift(lag)

        # Add lagged technical indicators
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
        df[f'bb_position_lag_{lag}'] = df['bb_position'].shift(lag)

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


def prepare_features_and_targets(df, target_col='next_close_change_pct'):
    """
    Prepare features and target variables
    """
    # Drop unnecessary columns
    drop_cols = ['time', 'arizona_time', 'next_close'] + [f'close_future_{i}' for i in range(2, 6)]
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
    target_cols = ['next_close_change_pct', 'next_direction'] + [f'change_future_{i}_pct' for i in range(2, 6)]
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
    Create sequences for LSTM/GRU models
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    return np.array(X_seq), np.array(y_seq)


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


def hyperparameter_grid_search(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape):
    """
    Simple hyperparameter grid search (simpler alternative to Bayesian Optimization)
    """
    # Define hyperparameter grid
    hyperparams = {
        'model_type': ['lstm', 'gru', 'bidirectional'],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'complexity': ['low', 'medium']
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

                    if model_type == 'lstm':
                        model = build_lstm_model(
                            input_shape=input_shape,
                            complexity=complexity,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                    elif model_type == 'gru':
                        model = build_gru_model(
                            input_shape=input_shape,
                            complexity=complexity,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                    elif model_type == 'bidirectional':
                        model = build_bidirectional_model(
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


def evaluate_model(model, X_test_seq, y_test_seq):
    """
    Evaluate the model on test data
    """
    # Predictions
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


def plot_results(history, test_results, feature_list):
    """
    Plot training history and test results
    """
    # Training history
    plt.figure(figsize=(20, 10))

    # Plot loss
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot MAE
    plt.subplot(2, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

    # Plot directional accuracy
    plt.subplot(2, 3, 3)
    plt.plot(history.history['directional_accuracy'], label='Training Dir. Accuracy')
    plt.plot(history.history['val_directional_accuracy'], label='Validation Dir. Accuracy')
    plt.title('Directional Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot predictions vs actual
    plt.subplot(2, 3, 4)
    plt.plot(test_results['actual'], label='Actual', alpha=0.7)
    plt.plot(test_results['predictions'], label='Predictions', alpha=0.7)
    plt.title('Test Set: Predictions vs Actual')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Value')

    # Plot prediction error
    plt.subplot(2, 3, 5)
    error = test_results['actual'] - test_results['predictions'].flatten()
    plt.plot(error)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Prediction Error')
    plt.xlabel('Sample')
    plt.ylabel('Error')

    # Plot scatter plot
    plt.subplot(2, 3, 6)
    plt.scatter(test_results['actual'], test_results['predictions'])
    plt.axline([0, 0], [1, 1], color='r', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

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


def save_model_and_metadata(model, scaler, feature_list, best_params, test_metrics):
    """
    Save model, scaler, feature list, and metadata
    """
    # Save model
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))

    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature list
    with open(os.path.join(MODEL_DIR, 'feature_list.pkl'), 'wb') as f:
        pickle.dump(feature_list, f)

    # Save metadata
    metadata = {
        'hyperparameters': best_params,
        'test_metrics': {k: float(v) if isinstance(v, np.float32) else v
                         for k, v in test_metrics.items()
                         if k not in ['predictions', 'actual']},
        'feature_count': len(feature_list),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=4)

    logger.info(f"Model and metadata saved to {MODEL_DIR}")


def main():
    # MT5 connection params
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    # Connect to MT5
    if not connect_to_mt5(account, password, server):
        return

    try:
        # Define date range for historical data (3 years)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=3 * 365)

        # Get historical data
        df = get_historical_data(SYMBOL, TIMEFRAME, start_date, end_date)
        if df is None:
            return

        # Filter for 5 PM Arizona time
        df_5pm = filter_5pm_data(df)

        # Feature engineering
        logger.info("Adding datetime features...")
        df_5pm = add_datetime_features(df_5pm)

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

        # Train final model
        logger.info("Training final model with optimal hyperparameters...")
        final_model, history = train_final_model(
            best_model,
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            batch_size=32,
            epochs=100
        )

        # Evaluate on test data
        logger.info("Evaluating final model on test data...")
        test_results = evaluate_model(final_model, X_test_seq, y_test_seq)

        # Calculate feature importance
        logger.info("Calculating feature importance...")
        feature_importance = calculate_feature_importance(
            final_model, X_test_seq, y_test_seq, feature_list
        )
        test_results['feature_importance'] = feature_importance

        # Plot results
        logger.info("Plotting results...")
        plot_results(history, test_results, feature_list)

        # Save model and metadata
        logger.info("Saving model and metadata...")
        save_model_and_metadata(final_model, scaler, feature_list, best_params, test_results)

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