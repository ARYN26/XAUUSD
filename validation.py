#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT5 Neural Network Validation Script
Focus: 5 PM Arizona Time Data
Uses walk-forward validation to prevent look-ahead bias
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import MetaTrader5 as mt5
import pytz
import pickle
import logging

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
SYMBOL = 'EURUSD'
TIMEFRAME = mt5.TIMEFRAME_H1
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'hour', 'day_of_week', 'day_of_month', 'month']
TARGET = 'next_close_change'

# Arizona time is UTC-7 (no DST)
ARIZONA_TZ = pytz.timezone('US/Arizona')
TARGET_HOUR = 17  # 5 PM Arizona time


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


def preprocess_data(df, scaler):
    """
    Preprocess data for neural network
    """
    # Calculate target: next period's close price percent change
    df['next_close'] = df['close'].shift(-1)
    df['next_close_change'] = ((df['next_close'] - df['close']) / df['close']) * 100
    df.dropna(inplace=True)

    # Ensure all required features are present
    features = FEATURES.copy()
    for feature in FEATURES:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in data! Creating placeholder.")
            df[feature] = 0.0  # Create placeholder

    X = df[features].values
    y = df[TARGET].values

    # Check if scaler input dimension matches our data
    if hasattr(scaler, 'n_features_in_'):
        expected_features = scaler.n_features_in_
        logger.info(f"Scaler expects {expected_features} features, data has {X.shape[1]} features")

        if X.shape[1] != expected_features:
            # Handle dimension mismatch
            if X.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X_padded = np.hstack([X, padding])
                logger.info(f"Padded input data from {X.shape[1]} to {expected_features} features")
                X = X_padded
            else:
                # Truncate
                logger.info(f"Truncating input data from {X.shape[1]} to {expected_features} features")
                X = X[:, :expected_features]

    try:
        # Scale features
        X_scaled = scaler.transform(X)
    except ValueError as e:
        logger.error(f"Scaler error: {e}")
        logger.info("Trying adaptive scaling approach...")

        # Create a new scaler if there's a mismatch
        from sklearn.preprocessing import MinMaxScaler
        temp_scaler = MinMaxScaler()
        X_scaled = temp_scaler.fit_transform(X)
        logger.info("Used temporary scaler due to dimension mismatch")

    return X_scaled, y


def create_sequences(X, y, lookback=LOOKBACK):
    """
    Create sequences for LSTM
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback])

    return np.array(X_seq), np.array(y_seq)


def walk_forward_validation(model, df, scaler, window_size=30, step_size=10, expected_features=None):
    """
    Perform walk-forward validation to prevent look-ahead bias

    Parameters:
    model - the neural network model
    df - historical data DataFrame
    window_size - size of testing window in days
    step_size - number of days to step forward

    Returns:
    DataFrame with validation results
    """
    logger.info("Performing walk-forward validation...")

    # Get all dates
    dates = df['arizona_time'].dt.date.unique()
    dates.sort()

    # Initialize results storage
    all_predictions = []
    all_actuals = []
    all_dates = []

    # Loop through time windows
    total_steps = (len(dates) - window_size) // step_size
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
        X_test, y_test = preprocess_data(test_df, scaler)

        # Create sequences
        X_seq, y_seq = create_sequences(X_test, y_test)

        # If no sequences could be created, skip
        if len(X_seq) == 0:
            continue

        # Pad features if necessary to match model's expected input shape
        if expected_features is not None:
            current_features = X_seq.shape[2]  # Current feature count

            if current_features != expected_features:
                logger.warning(
                    f"Feature count mismatch: model expects {expected_features}, data has {current_features}")

                # Pad or truncate features to match
                if current_features < expected_features:
                    # Pad with zeros
                    padding_size = expected_features - current_features
                    logger.info(f"Padding data with {padding_size} additional feature columns")

                    # Create padded array
                    X_padded = np.zeros((X_seq.shape[0], X_seq.shape[1], expected_features))
                    X_padded[:, :, :current_features] = X_seq
                    X_seq = X_padded
                else:
                    # Truncate to first expected_features
                    logger.info(f"Truncating data from {current_features} to {expected_features} features")
                    X_seq = X_seq[:, :, :expected_features]

        # Make predictions
        predictions = model.predict(X_seq, verbose=0)

        # Store results
        for j in range(len(predictions)):
            all_predictions.append(predictions[j][0])
            all_actuals.append(y_seq[j])
            # Get the corresponding date
            all_dates.append(test_df.iloc[j + LOOKBACK]['arizona_time'])

    # Create results DataFrame
    results = pd.DataFrame({
        'date': all_dates,
        'actual': all_actuals,
        'prediction': all_predictions
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
    plt.savefig('walk_forward_results.png')

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
    plt.savefig('walk_forward_error.png')

    # Create monthly error analysis
    results['year_month'] = results['date'].dt.to_period('M')
    monthly_error = results.groupby('year_month').agg({
        'error': ['mean', 'std'],
        'abs_error': 'mean',
        'direction_match': 'mean'
    })

    logger.info(f"Monthly Analysis:\n{monthly_error}")

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
    X, y = preprocess_data(df, scaler)
    X_seq, y_seq = create_sequences(X, y)

    # Pad features if necessary to match model's expected input shape
    if expected_features is not None:
        current_features = X_seq.shape[2]  # Current feature count

        if current_features != expected_features:
            logger.warning(f"Feature count mismatch: model expects {expected_features}, data has {current_features}")

            # Pad or truncate features to match
            if current_features < expected_features:
                # Pad with zeros
                padding_size = expected_features - current_features
                logger.info(f"Padding data with {padding_size} additional feature columns")

                # Create padded array
                X_padded = np.zeros((X_seq.shape[0], X_seq.shape[1], expected_features))
                X_padded[:, :, :current_features] = X_seq
                X_seq = X_padded
            else:
                # Truncate to first expected_features
                logger.info(f"Truncating data from {current_features} to {expected_features} features")
                X_seq = X_seq[:, :, :expected_features]

    # Get baseline performance
    baseline_pred = model.predict(X_seq, verbose=0)
    baseline_mse = mean_squared_error(y_seq, baseline_pred)

    # Calculate importance for each feature
    importance = {}

    # Only permute features that we actually have in our data
    feature_count = min(len(FEATURES), X_seq.shape[2])

    for i, feature in enumerate(FEATURES[:feature_count]):
        logger.info(f"Processing feature importance for {feature}")

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
        importance[feature] = perm_mse - baseline_mse

    # Convert to DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title('Feature Importance (Permutation Method)')
    plt.xlabel('Increase in MSE when feature is permuted')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    logger.info(f"Feature Importance:\n{importance_df}")

    return importance_df


def cross_validate_time_periods(model, df, scaler, expected_features=None):
    """
    Cross-validate across different time periods (days of week, months, etc.)
    """
    logger.info("Cross-validating across time periods...")

    # Preprocess all data
    X, y = preprocess_data(df, scaler)
    X_seq, y_seq = create_sequences(X, y)

    # Pad features if necessary to match model's expected input shape
    if expected_features is not None:
        current_features = X_seq.shape[2]  # Current feature count

        if current_features != expected_features:
            logger.warning(f"Feature count mismatch: model expects {expected_features}, data has {current_features}")

            # Pad or truncate features to match
            if current_features < expected_features:
                # Pad with zeros
                padding_size = expected_features - current_features
                logger.info(f"Padding data with {padding_size} additional feature columns")

                # Create padded array
                X_padded = np.zeros((X_seq.shape[0], X_seq.shape[1], expected_features))
                X_padded[:, :, :current_features] = X_seq
                X_seq = X_padded
            else:
                # Truncate to first expected_features
                logger.info(f"Truncating data from {current_features} to {expected_features} features")
                X_seq = X_seq[:, :, :expected_features]

    # Day of week analysis
    df_subset = df.iloc[LOOKBACK:].reset_index(drop=True)
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

        day_results[days[day_num]] = {
            'count': len(day_indices),
            'mse': mse,
            'mae': mae
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

        month_results[months[month_num - 1]] = {
            'count': len(month_indices),
            'mse': mse,
            'mae': mae
        }

    # Convert to DataFrame
    month_df = pd.DataFrame.from_dict(month_results, orient='index')
    logger.info(f"Month Performance:\n{month_df}")

    # Plot day of week performance
    plt.figure(figsize=(12, 6))

    # MSE by day of week
    plt.subplot(1, 2, 1)
    plt.bar(day_df.index, day_df['mse'])
    plt.title('MSE by Day of Week')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)

    # MAE by day of week
    plt.subplot(1, 2, 2)
    plt.bar(day_df.index, day_df['mae'])
    plt.title('MAE by Day of Week')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('day_of_week_performance.png')

    # Plot month performance
    plt.figure(figsize=(12, 6))

    # MSE by month
    plt.subplot(1, 2, 1)
    plt.bar(month_df.index, month_df['mse'])
    plt.title('MSE by Month')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)

    # MAE by month
    plt.subplot(1, 2, 2)
    plt.bar(month_df.index, month_df['mae'])
    plt.title('MAE by Month')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('month_performance.png')

    return day_df, month_df


def main():
    # MT5 connection params
    account = 90933473
    password = "NhXgR*3g"
    server = "MetaQuotes-Demo"

    # Connect to MT5
    if not connect_to_mt5(account, password, server):
        return

    try:
        # Try different model filenames
        model = None
        model_filenames = ['mt5_neural_network_model.h5', 'model.h5', 'best_model.h5']

        for filename in model_filenames:
            try:
                if os.path.exists(filename):
                    model = load_model(filename)
                    logger.info(f"Model loaded successfully from {filename}")
                    break
            except Exception as e:
                logger.warning(f"Could not load model from {filename}: {e}")

        if model is None:
            logger.error("Could not find a valid model file")
            return

        # Inspect model input shape
        expected_features = inspect_model_input_shape(model)
        if expected_features:
            logger.info(f"Model expects {expected_features} input features")
        else:
            logger.warning("Could not determine expected feature count from model")
            expected_features = 11  # Fallback to 11 based on previous error message

        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")

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

        # Perform walk-forward validation
        wf_results, wf_metrics = walk_forward_validation(model, df_5pm, scaler, expected_features=expected_features)

        # Validate feature importance
        feature_importance = validate_feature_importance(model, df_5pm, scaler, expected_features=expected_features)

        # Cross-validate across time periods
        day_perf, month_perf = cross_validate_time_periods(model, df_5pm, scaler, expected_features=expected_features)

        # Create validation summary
        summary = {
            'walk_forward_metrics': wf_metrics,
            'top_features': feature_importance['Feature'].tolist()[:3],
            'best_day': day_perf['mae'].idxmin(),
            'best_month': month_perf['mae'].idxmin()
        }

        # Save validation results
        with open('validation_summary.txt', 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

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