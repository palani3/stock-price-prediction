# app/routers/chart.py
from fastapi import APIRouter, HTTPException
from app.models.stock import StockResponse, TrainingResponse
from app.services.stock_service import (
    fetch_stock_data,
    prepare_stock_data,
    prepare_features,
    build_model,
    generate_signals
)
from sklearn.model_selection import train_test_split
import tensorflow as tf

router = APIRouter()

@router.post("/train/{symbol}", response_model=TrainingResponse)
async def train_model(symbol: str, interval: str = "1d"):
    """
    Train model using fetched stock data.
    
    Parameters:
        symbol: Stock symbol (e.g., 'TCS', 'INFY')
        interval: Data interval ('1d', '1wk', '1mo')
        
    Returns:
        TrainingResponse object containing model predictions and analysis
    """
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(symbol, interval)
        
        # Prepare data with technical indicators
        df = prepare_stock_data(stock_data)
        
        # Prepare features for training
        X, y, scaler = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Build and train model
        model = build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train with early stopping
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Make prediction for latest data
        last_sequence = X[-1:]
        prediction = model.predict(last_sequence)
        latest_signal = 'buy' if prediction[0][0] > 0.5 else 'sell'
        confidence = float(prediction[0][0] if latest_signal == 'buy' else 1 - prediction[0][0])
        
        # Generate trading signals
        trading_signals = generate_signals(df)
        
        return TrainingResponse(
            model_accuracy=float(history.history['accuracy'][-1]),
            validation_accuracy=float(history.history['val_accuracy'][-1]),
            latest_prediction=latest_signal,
            prediction_confidence=confidence,
            support_level=float(df['support'].iloc[-1]),
            resistance_level=float(df['resistance'].iloc[-1]),
            trading_signals=trading_signals
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{symbol}")
async def get_stock_data(symbol: str, interval: str = "1d"):
    """
    Get stock data with technical indicators.
    
    Parameters:
        symbol: Stock symbol (e.g., 'TCS', 'INFY')
        interval: Data interval ('1d', '1wk', '1mo')
        
    Returns:
        StockResponse object containing the stock data
    """
    try:
        return fetch_stock_data(symbol, interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))