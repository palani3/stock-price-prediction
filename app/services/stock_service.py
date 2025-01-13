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
    Fetches complete historical stock data and today's data.
    """
    try:
        print(f"Fetching data for symbol: {symbol}, interval: {interval}")

        # Validate interval
        valid_intervals = ["1d", "1wk", "1mo"]
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Please use one of: {', '.join(valid_intervals)}"
            )

        # Special handling for NIFTY 50 index
        if symbol.upper() in ["^NSEI", "NSEI"]:
            fetch_symbol = "^NSEI"  # Use ^NSEI for NIFTY 50
        else:
            # Handle case sensitivity for regular NSE stocks
            if not symbol.endswith('.NS'):
                fetch_symbol = f"{symbol.upper()}.NS"
            else:
                base_symbol = symbol[:-3]
                fetch_symbol = f"{base_symbol.upper()}.NS"

        print(f"Processed symbol: {fetch_symbol}")

        # Set date range
        if interval == "1d":
            start_date = '2010-01-01'
        else:
            start_date = '1990-01-01'
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch historical data
        try:
            stock_data = yf.download(
                fetch_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            print(f"Historical data fetched. Shape: {stock_data.shape}")
            
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
                detail=f"No data found for {fetch_symbol} with interval {interval}"
            )

        # Process historical data
        stock_data.reset_index(inplace=True)
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]

        # Validate columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # Process historical data points
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

        # Fetch today's data
        try:
            stock = yf.Ticker(fetch_symbol)  # Use fetch_symbol here too
            today_data = stock.history(period='1d', interval='1d')
            
            if not today_data.empty:
                today = datetime.now().strftime('%Y-%m-%d')
                today_row = today_data.iloc[-1]
                
                today_point = StockDatas(
                    date=today,
                    open=round(float(today_row['Open']), 2),
                    close=round(float(today_row['Close']), 2),
                    high=round(float(today_row['High']), 2),
                    low=round(float(today_row['Low']), 2),
                    volume=int(today_row['Volume'])
                )
                
                # Add today's data if it's not already in the processed data
                if not processed_data or processed_data[-1].date != today:
                    processed_data.append(today_point)
                    print("Added today's data successfully")

        except Exception as today_error:
            print(f"Error fetching today's data: {str(today_error)}")
            # Continue without today's data if there's an error

        if not processed_data:
            raise HTTPException(
                status_code=404,
                detail="Could not process any data points"
            )

        print(f"Successfully processed {len(processed_data)} data points")

        # Add metadata
        first_date = processed_data[0].date if processed_data else None
        last_date = processed_data[-1].date if processed_data else None
        
        # Use original symbol for response
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
    
    # Get key price levels and signals
    buy_prices = df.loc[signals['final_signal'] == 'buy', 'close'].iloc[-5:].round(2)
    sell_prices = df.loc[signals['final_signal'] == 'sell', 'close'].iloc[-5:].round(2)
    
    signals_text = []
    for date, signal in signals['final_signal'].tail(5).items():
        price = round(df['close'].loc[date], 2)
        signals_text.append(f"{date.strftime('%Y-%m-%d')}: {signal} at {price}")
    
    buy_signals = [f"{date.strftime('%Y-%m-%d')}: {price}" 
                  for date, price in buy_prices.items()]
    sell_signals = [f"{date.strftime('%Y-%m-%d')}: {price}" 
                   for date, price in sell_prices.items()]
    
    return {
        'signals': '\n'.join(signals_text),
        'buy_points': '\n'.join(buy_signals) if buy_signals else 'No recent buy signals',
        'sell_points': '\n'.join(sell_signals) if sell_signals else 'No recent sell signals'
    }
    
    return last_signals