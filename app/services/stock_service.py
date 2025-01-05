# app/services/stock_service.py
from fastapi import HTTPException
from app.models.stock import StockDatas, StockResponse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import ta

def fetch_stock_data(symbol: str, interval: str = "1d") -> StockResponse:
    """
    Fetches complete historical stock data for daily, weekly, or monthly intervals.
    
    Parameters:
        symbol: Stock symbol (e.g., 'ITC' or 'ITC.NS')
        interval: Time interval - can be '1d' (daily), '1wk' (weekly), or '1mo' (monthly)
        
    Returns:
        StockResponse object containing the processed data and metadata
    """
    try:
        # First, let's add some debug logging to understand what's happening
        print(f"Fetching data for symbol: {symbol}, interval: {interval}")

        # Validate the interval first
        valid_intervals = ["1d", "1wk", "1mo"]
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Please use one of: {', '.join(valid_intervals)}"
            )

        # Handle case sensitivity for NSE stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol.upper()}.NS"
        else:
            base_symbol = symbol[:-3]
            symbol = f"{base_symbol.upper()}.NS"

        print(f"Processed symbol: {symbol}")

        # For daily data, let's start from a more recent date to ensure we get data
        # Many NSE stocks might not have data from 1990
        if interval == "1d":
            start_date = '2010-01-01'  # Starting from 2010 instead of 1990
        else:
            start_date = '1990-01-01'
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_date} to {end_date}")

        # Add more error checking for the download
        try:
            stock_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True  # Added this to handle stock splits and dividends
            )
            
            print(f"Data fetched. Shape: {stock_data.shape}")
            
        except Exception as download_error:
            print(f"Download error: {str(download_error)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download data: {str(download_error)}"
            )

        if stock_data.empty:
            print("No data received from Yahoo Finance")
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol} with interval {interval}"
            )

        # Process the downloaded data
        stock_data.reset_index(inplace=True)
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]

        # Add a check for required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            print(f"Available columns: {stock_data.columns.tolist()}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # Create the list of processed data points
        processed_data = []
        for _, row in stock_data.iterrows():
            try:
                date_str = row['Date'].strftime('%Y-%m-%d')
                
                data_point = StockDatas(
                    date=date_str,
                    open=round(float(row['Open']), 2),
                    close=round(float(row['Close']), 2),
                    high=round(float(row['High']), 2),
                    low=round(float(row['Low']), 2),
                    volume=int(row['Volume'])
                )
                processed_data.append(data_point)
            except Exception as row_error:
                print(f"Error processing row: {row}")
                print(f"Error details: {str(row_error)}")
                continue

        if not processed_data:
            raise HTTPException(
                status_code=404,
                detail="Could not process any data points"
            )

        print(f"Successfully processed {len(processed_data)} data points")

        # Add metadata about the data range
        first_date = processed_data[0].date if processed_data else None
        last_date = processed_data[-1].date if processed_data else None
        
        return StockResponse(
            symbol=symbol,
            interval=interval,
            data=processed_data,
            start_date=first_date,
            end_date=last_date,
            total_records=len(processed_data)
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

def prepare_stock_data(stock_response: StockResponse) -> pd.DataFrame:
    """Prepare stock data for analysis"""
    df = pd.DataFrame([{
        'date': data.date,
        'open': data.open,
        'high': data.high,
        'low': data.low,
        'close': data.close,
        'volume': data.volume
    } for data in stock_response.data])
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(df['close'], window=21)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    
    df['sma_21'] = ta.trend.sma_indicator(df['close'], window=21)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    
    df['close_pct_change'] = df['close'].pct_change().clip(-0.2, 0.2)
    df['volume_pct_change'] = df['volume'].pct_change().clip(-0.5, 0.5)
    
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    return df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

def prepare_features(df: pd.DataFrame, sequence_length: int = 20):
    """Prepare features for LSTM model"""
    try:
        features = [
            'close_pct_change',
            'volume_pct_change',
            'rsi',
            'macd',
            'bb_high',
            'bb_low',
            'sma_21',
            'sma_200'
        ]
        
        feature_df = df[features].copy()
        feature_df['price_momentum'] = df['close'].pct_change(5).fillna(0)
        feature_df['volume_momentum'] = df['volume'].pct_change(5).fillna(0)
        feature_df['volatility'] = df['close'].rolling(window=10).std().fillna(0)
        
        sma_diff = (df['sma_21'] - df['sma_200'])
        sma_denominator = df['sma_200'].replace(0, np.nan)
        trend_strength = (sma_diff / sma_denominator * 100).fillna(0)
        feature_df['trend_strength'] = trend_strength
        
        # Clean and scale data
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        for col in feature_df.columns:
            mean_val = feature_df[col].mean()
            std_val = feature_df[col].std()
            feature_df[col] = feature_df[col].clip(
                lower=mean_val - 3 * std_val,
                upper=mean_val + 3 * std_val
            )
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_df)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - 5):
            X.append(scaled_data[i - sequence_length:i])
            current_price = float(df['close'].iloc[i])
            future_price = float(df['close'].iloc[i + 5])
            future_return = (future_price - current_price) / current_price if current_price > 0 else 0
            y.append(1 if future_return > 0.01 else 0)
        
        return np.array(X), np.array(y), scaler
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing features: {str(e)}")

def build_model(input_shape):
    """Create LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_signals(df: pd.DataFrame) -> dict:
    """Generate trading signals"""
    signals = pd.DataFrame(index=df.index)
    
    # Generate signals
    signals['rsi_signal'] = np.where(df['rsi'] < 30, 'buy',
                                   np.where(df['rsi'] > 70, 'sell', 'hold'))
    signals['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 'buy', 'sell')
    signals['bb_signal'] = np.where(df['close'] < df['bb_low'], 'buy',
                                  np.where(df['close'] > df['bb_high'], 'sell', 'hold'))
    
    signals['final_signal'] = signals.mode(axis=1)[0]
    
    # Convert the last 5 signals to a dict with string dates
    last_signals = signals['final_signal'].tail(5)
    return {
        date.strftime('%Y-%m-%d'): signal 
        for date, signal in last_signals.items()
    }