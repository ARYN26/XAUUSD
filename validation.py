import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import matplotlib.pyplot as plt


# Test connection to MetaTrader 5
def test_connection():
    # Account credentials
    login = 90933473
    password = "NhXgR*3g"

    if not mt5.initialize():
        print("initialize() failed")
        return False

    # Connect to account
    authorized = mt5.login(login, password, server="MetaQuotes-Demo")
    if not authorized:
        print(f"Failed to connect to account {login}, error code: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"Connected to account {login}")

    # Print account info
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"Account balance: {account_info.balance}")
        print(f"Account equity: {account_info.equity}")

    return True


# Get data and display available columns
def test_data_retrieval():
    # Get some recent data
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1

    # Get 100 bars
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)

    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}, error code: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Display information about the DataFrame
    print("\nDataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df


# Calculate RSI for testing
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Test technical indicators calculation
def test_indicators(df):
    if df is None:
        return

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    print("\nCalculating technical indicators...")

    # Create technical indicators
    df_copy['rsi'] = calculate_rsi(df_copy['close'], 14)
    df_copy['ma_20'] = df_copy['close'].rolling(window=20).mean()
    df_copy['ma_50'] = df_copy['close'].rolling(window=50).mean()
    df_copy['ma_200'] = df_copy['close'].rolling(window=200).mean()
    df_copy['volatility'] = df_copy['close'].rolling(window=20).std()

    # Check for NaN values
    nan_count = df_copy.isna().sum().sum()
    print(f"Total NaN values after indicator calculation: {nan_count}")

    # Drop rows with NaN values
    df_copy.dropna(inplace=True)
    print(f"Shape after dropping NaN rows: {df_copy.shape}")

    # Plot some of the indicators
    plt.figure(figsize=(12, 10))

    # Plot 1: Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df_copy.index, df_copy['close'], label='Close Price')
    plt.plot(df_copy.index, df_copy['ma_20'], label='20-period MA')
    plt.plot(df_copy.index, df_copy['ma_50'], label='50-period MA')
    plt.title('Price and Moving Averages')
    plt.legend()

    # Plot 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(df_copy.index, df_copy['rsi'])
    plt.axhline(y=70, color='r', linestyle='-')
    plt.axhline(y=30, color='g', linestyle='-')
    plt.title('RSI')

    # Plot 3: Volatility
    plt.subplot(3, 1, 3)
    plt.plot(df_copy.index, df_copy['volatility'])
    plt.title('Volatility (20-period Std Dev)')

    plt.tight_layout()
    plt.savefig('technical_indicators.png')
    print("Saved technical indicators plot to 'technical_indicators.png'")

    return df_copy


# Test scaler creation
def test_scaler(df):
    if df is None:
        return None

    print("\nTesting scaler creation...")

    # Get available columns
    base_features = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
    available_features = [col for col in base_features if col in df.columns]

    # Add technical indicators to features list
    all_features = available_features + ['rsi', 'ma_20', 'ma_50', 'ma_200', 'volatility']

    # Check if all features exist in the DataFrame
    missing_features = [feat for feat in all_features if feat not in df.columns]
    if missing_features:
        print(f"Warning: These features are missing: {missing_features}")
        all_features = [feat for feat in all_features if feat in df.columns]

    print(f"Using features for scaler: {all_features}")

    # Create and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[all_features])

    print(f"Scaled data shape: {scaled_data.shape}")

    # Save the scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Saved scaler to 'scaler.pkl'")

    return scaler


# Test Arizona time check
def test_arizona_time():
    print("\nTesting Arizona time check...")

    try:
        # Get current time in Arizona
        arizona_tz = pytz.timezone("America/Phoenix")
        current_time = datetime.now(arizona_tz)

        print(f"Current time in Arizona: {current_time}")
        print(f"Current hour in Arizona: {current_time.hour}")

        # Check if it's 5 PM
        is_five = current_time.hour == 17
        print(f"Is it 5 PM in Arizona? {is_five}")

        # Show when the next 5 PM will be
        if current_time.hour < 17:
            next_five = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            next_five = current_time.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)

        time_diff = next_five - current_time
        print(f"Next 5 PM in Arizona will be in: {time_diff}")

    except Exception as e:
        print(f"Error checking Arizona time: {e}")
        # Fallback to UTC calculation
        current_time_utc = datetime.now(pytz.UTC)
        print(f"Current UTC time: {current_time_utc}")
        # Arizona is UTC-7 (no DST)
        arizona_hour = (current_time_utc.hour - 7) % 24
        print(f"Estimated Arizona hour (using UTC-7): {arizona_hour}")


# Main test function
def main():
    print("Starting MetaTrader 5 Test Script...")

    # Test connection
    if not test_connection():
        print("Connection test failed. Exiting.")
        return

    # Test data retrieval
    df = test_data_retrieval()

    # Test indicators
    df_with_indicators = test_indicators(df)

    # Test scaler
    scaler = test_scaler(df_with_indicators)

    # Test Arizona time
    test_arizona_time()

    # Clean up
    mt5.shutdown()
    print("\nTest script completed.")


if __name__ == "__main__":
    main()