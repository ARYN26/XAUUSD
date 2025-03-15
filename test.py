#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT5 Neural Network Testing Script - Final Version
Focus: 5 PM Arizona Time Data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MetaTrader5 as mt5
import pytz
import pickle
import logging
import io
import re
from contextlib import redirect_stdout

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
LOOKBACK = 20
SYMBOL = 'EURUSD'
TIMEFRAME = mt5.TIMEFRAME_H1

# Paths
MODEL_DIR = 'models'
TEST_RESULTS_DIR = 'test_results'
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time


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

        if 'close_diff' in df.columns:
            df[f'close_diff_lag_{lag}'] = df['close_diff'].shift(lag)

        if 'close_diff_pct' in df.columns:
            df[f'close_diff_pct_lag_{lag}'] = df['close_diff_pct'].shift(lag)

        # Add lagged technical indicators
        if 'rsi_14' in df.columns:
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)

        if 'macd' in df.columns:
            df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

        if 'bb_position' in df.columns:
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


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM/GRU models
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    return np.array(X_seq), np.array(y_seq)


def evaluate_model():
    """
    Evaluate the trained model on test data
    """
    # Try three different methods to load the model
    model = None

    # Method 1: Load with compile=False and recompile
    try:
        model_path = os.path.join(MODEL_DIR, 'final_model.h5')
        logger.info(f"Method 1: Loading model without compilation...")
        model = load_model(model_path, compile=False)

        # Recompile with custom metrics
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[mae_custom, r2_keras, directional_accuracy]
        )
        logger.info(f"Model loaded and recompiled successfully")

    except Exception as e:
        logger.error(f"Method 1 failed: {e}")

        # Method 2: Using custom_objects dictionary
        try:
            logger.info(f"Method 2: Loading with custom objects...")
            custom_objects = {
                'r2_keras': r2_keras,
                'directional_accuracy': directional_accuracy,
                'mse': mse_custom,
                'mae': mae_custom
            }

            model = load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded with custom objects")
        except Exception as e:
            logger.error(f"Method 2 failed: {e}")

            # Method 3: Create fresh model with same architecture
            try:
                logger.info(f"Method 3: Loading model weights only...")

                # Get the model architecture from the saved file
                if os.path.exists(os.path.join(MODEL_DIR, 'model_architecture.json')):
                    with open(os.path.join(MODEL_DIR, 'model_architecture.json'), 'r') as f:
                        import json
                        model_json = json.load(f)

                    from tensorflow.keras.models import model_from_json
                    model = model_from_json(model_json)
                    model.load_weights(model_path)

                    # Compile the model
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=[mae_custom, r2_keras, directional_accuracy]
                    )
                    logger.info(f"Model loaded from architecture and weights")
                else:
                    logger.error("Model architecture file not found, cannot proceed")
                    return None
            except Exception as e:
                logger.error(f"Method 3 failed: {e}")
                logger.error("All model loading methods failed, cannot proceed with evaluation")
                return None

    if model is None:
        logger.error("Failed to load model using any method")
        return None

    # Extract expected feature count
    expected_features = inspect_model_input_shape(model)
    if expected_features:
        logger.info(f"Model expects {expected_features} input features")
    else:
        logger.warning("Could not determine expected feature count, using default")
        expected_features = 103  # Based on the enhanced training logs

    # Load feature list
    try:
        with open(os.path.join(MODEL_DIR, 'feature_list.pkl'), 'rb') as f:
            feature_list = pickle.load(f)
        logger.info(f"Loaded {len(feature_list)} features from feature list")
    except:
        logger.warning("Could not load feature list, will create features from scratch")
        feature_list = None

    # Load scaler
    try:
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
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
            logger.error(f"Not enough data points at 5 PM AZ time (only {len(df_5pm)} found)")
            return

        # Add features to match the features used in training
        logger.info("Adding datetime features...")
        df_5pm = add_datetime_features(df_5pm)

        logger.info("Adding technical indicators...")
        df_5pm = add_technical_indicators(df_5pm)

        logger.info("Adding lagged features...")
        df_5pm = add_lagged_features(df_5pm)

        logger.info("Adding target variables...")
        df_5pm = add_target_variables(df_5pm)

        # Prepare features and target
        X, y, actual_feature_list = prepare_features_and_targets(df_5pm)

        # Scale features
        X_scaled = scaler.transform(X)

        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y)

        # Adjust feature dimensions if needed
        if expected_features:
            current_features = X_seq.shape[2]
            if current_features != expected_features:
                logger.warning(
                    f"Feature count mismatch: model expects {expected_features}, data has {current_features}")

                # Pad or truncate features to match
                if current_features < expected_features:
                    padding_size = expected_features - current_features
                    logger.info(f"Padding data with {padding_size} additional feature columns")

                    # Create padded array
                    X_padded = np.zeros((X_seq.shape[0], X_seq.shape[1], expected_features))
                    X_padded[:, :, :current_features] = X_seq
                    X_seq = X_padded
                else:
                    logger.info(f"Truncating data from {current_features} to {expected_features} features")
                    X_seq = X_seq[:, :, :expected_features]

        logger.info(f"Test data shape: {X_seq.shape}")

        # Make predictions
        y_pred = model.predict(X_seq)

        # Calculate metrics
        mse = mean_squared_error(y_seq, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_seq, y_pred)
        r2 = r2_score(y_seq, y_pred)

        # Directional accuracy
        directional_acc = np.mean((np.sign(y_seq) == np.sign(y_pred)).astype(int))

        logger.info(f"Test Metrics:")
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.6f}")
        logger.info(f"Directional Accuracy: {directional_acc:.2%}")

        # Plot predictions vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(y_seq, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title('Model Predictions vs Actual Values')
        plt.xlabel('Time Step')
        plt.ylabel('Price Change (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'test_predictions.png'))

        # Plot prediction error
        plt.figure(figsize=(14, 7))
        prediction_error = y_seq - y_pred.flatten()
        plt.plot(prediction_error)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Prediction Error')
        plt.xlabel('Time Step')
        plt.ylabel('Error')
        plt.grid(True)
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'prediction_error.png'))

        # Plot scatter of predicted vs actual
        plt.figure(figsize=(10, 10))
        plt.scatter(y_seq, y_pred)
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # Add 45-degree line
        min_val = min(np.min(y_seq), np.min(y_pred))
        max_val = max(np.max(y_seq), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.grid(True)
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'scatter_comparison.png'))

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_acc
        }

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    finally:
        mt5.shutdown()


def backtest_strategy(threshold=0.1):
    """
    Backtest the trained model strategy on historical data
    """
    # Try three different methods to load the model
    model = None

    # Method 1: Load with compile=False and recompile
    try:
        model_path = os.path.join(MODEL_DIR, 'final_model.h5')
        logger.info(f"Method 1: Loading model without compilation...")
        model = load_model(model_path, compile=False)

        # Recompile with custom metrics
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[mae_custom, r2_keras, directional_accuracy]
        )
        logger.info(f"Model loaded and recompiled successfully")

    except Exception as e:
        logger.error(f"Method 1 failed: {e}")

        # Method 2: Using custom_objects dictionary
        try:
            logger.info(f"Method 2: Loading with custom objects...")
            custom_objects = {
                'r2_keras': r2_keras,
                'directional_accuracy': directional_accuracy,
                'mse': mse_custom,
                'mae': mae_custom
            }

            model = load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded with custom objects")
        except Exception as e:
            logger.error(f"Method 2 failed: {e}")

            # Method 3: Create fresh model with same architecture
            try:
                logger.info(f"Method 3: Loading model weights only...")

                # Get the model architecture from the saved file
                if os.path.exists(os.path.join(MODEL_DIR, 'model_architecture.json')):
                    with open(os.path.join(MODEL_DIR, 'model_architecture.json'), 'r') as f:
                        import json
                        model_json = json.load(f)

                    from tensorflow.keras.models import model_from_json
                    model = model_from_json(model_json)
                    model.load_weights(model_path)

                    # Compile the model
                    model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=[mae_custom, r2_keras, directional_accuracy]
                    )
                    logger.info(f"Model loaded from architecture and weights")
                else:
                    logger.error("Model architecture file not found, cannot proceed")
                    return None
            except Exception as e:
                logger.error(f"Method 3 failed: {e}")
                logger.error("All model loading methods failed, cannot proceed with evaluation")
                return None

    if model is None:
        logger.error("Failed to load model using any method")
        return None

    # Extract expected feature count
    expected_features = inspect_model_input_shape(model)
    if expected_features:
        logger.info(f"Model expects {expected_features} input features")
    else:
        logger.warning("Could not determine expected feature count, using default")
        expected_features = 103  # Based on the enhanced training logs

    # Load scaler
    try:
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return

    # Connect to MT5
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    if not connect_to_mt5(account, password, server):
        return

    try:
        # Define date range for backtest (6 months)
        end_date = datetime.now(ARIZONA_TZ)
        start_date = end_date - timedelta(days=180)

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

        # Add features to match the features used in training
        logger.info("Adding datetime features...")
        df_5pm = add_datetime_features(df_5pm)

        logger.info("Adding technical indicators...")
        df_5pm = add_technical_indicators(df_5pm)

        logger.info("Adding lagged features...")
        df_5pm = add_lagged_features(df_5pm)

        logger.info("Adding target variables...")
        df_5pm = add_target_variables(df_5pm)

        # Prepare features and target
        X, y, feature_list = prepare_features_and_targets(df_5pm)

        # Scale features
        X_scaled = scaler.transform(X)

        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, y)

        # Adjust feature dimensions if needed
        if expected_features:
            current_features = X_seq.shape[2]
            if current_features != expected_features:
                logger.warning(
                    f"Feature count mismatch: model expects {expected_features}, data has {current_features}")

                # Pad or truncate features to match
                if current_features < expected_features:
                    padding_size = expected_features - current_features
                    logger.info(f"Padding data with {padding_size} additional feature columns")

                    # Create padded array
                    X_padded = np.zeros((X_seq.shape[0], X_seq.shape[1], expected_features))
                    X_padded[:, :, :current_features] = X_seq
                    X_seq = X_padded
                else:
                    logger.info(f"Truncating data from {current_features} to {expected_features} features")
                    X_seq = X_seq[:, :, :expected_features]

        logger.info(f"Backtest data shape: {X_seq.shape}")

        # Make predictions
        predictions = model.predict(X_seq)

        # Create results DataFrame with dates
        dates = df_5pm.iloc[LOOKBACK:]['arizona_time'].reset_index(drop=True)

        df_results = pd.DataFrame({
            'date': dates,
            'actual': y_seq.flatten(),
            'prediction': predictions.flatten()
        })

        # Set up trading strategy
        df_results['signal'] = 0  # 0 = no trade, 1 = buy, -1 = sell
        df_results.loc[df_results['prediction'] > threshold, 'signal'] = 1
        df_results.loc[df_results['prediction'] < -threshold, 'signal'] = -1

        # Calculate returns
        df_results['strategy_return'] = df_results['signal'] * df_results['actual']

        # Calculate cumulative returns
        df_results['cumulative_market_return'] = df_results['actual'].cumsum()
        df_results['cumulative_strategy_return'] = df_results['strategy_return'].cumsum()

        # Calculate metrics
        total_trades = (df_results['signal'] != 0).sum()
        winning_trades = (df_results['strategy_return'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        market_return = df_results['cumulative_market_return'].iloc[-1]
        strategy_return = df_results['cumulative_strategy_return'].iloc[-1]

        # Plot results
        plt.figure(figsize=(14, 7))
        plt.plot(df_results['date'], df_results['cumulative_market_return'], label='Buy and Hold', alpha=0.7)
        plt.plot(df_results['date'], df_results['cumulative_strategy_return'], label='Strategy', alpha=0.7)
        plt.title('Cumulative Returns: Strategy vs Buy and Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'backtest_results.png'))

        # Plot signals
        plt.figure(figsize=(14, 7))
        buy_signals = df_results[df_results['signal'] == 1]
        sell_signals = df_results[df_results['signal'] == -1]

        # Plot price
        plt.plot(df_results['date'], df_results['actual'].cumsum(), label='Price', alpha=0.7)

        # Plot buy/sell signals
        plt.scatter(buy_signals['date'], buy_signals['cumulative_market_return'],
                    marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(sell_signals['date'], sell_signals['cumulative_market_return'],
                    marker='v', color='red', s=100, label='Sell Signal')

        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Price Change (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, 'trading_signals.png'))

        # Log results
        logger.info(f"Backtest Results:")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winning trades: {winning_trades}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Market return: {market_return:.2f}%")
        logger.info(f"Strategy return: {strategy_return:.2f}%")

        # Save detailed backtest results
        df_results.to_csv(os.path.join(TEST_RESULTS_DIR, 'backtest_detailed.csv'))

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'market_return': market_return,
            'strategy_return': strategy_return
        }

    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    finally:
        mt5.shutdown()


def main():
    # Make sure model and scaler exist
    model_path = os.path.join(MODEL_DIR, 'final_model.h5')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return

    # Evaluate model
    logger.info("Evaluating model on test data...")
    metrics = evaluate_model()

    if metrics:
        # Run backtest with different thresholds
        thresholds = [0.05, 0.1, 0.2, 0.3]

        for threshold in thresholds:
            logger.info(f"Running backtest with threshold {threshold}...")
            backtest_metrics = backtest_strategy(threshold=threshold)

            if backtest_metrics:
                logger.info(f"Threshold {threshold} results:")
                logger.info(f"Total trades: {backtest_metrics['total_trades']}")
                logger.info(f"Win rate: {backtest_metrics['win_rate']:.2%}")
                logger.info(f"Strategy return: {backtest_metrics['strategy_return']:.2f}%")

    logger.info("Testing completed successfully")


if __name__ == "__main__":
    main()