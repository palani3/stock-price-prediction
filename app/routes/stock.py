from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import yfinance as yf
import asyncio
import json
from typing import Dict, List



from fastapi import APIRouter, HTTPException, Depends,WebSocket, WebSocketDisconnect
from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta  # Add timedelta here
from typing import Optional, List  # Add this import
from ..models.stock import (
    StockInfo,
    StockSearchResponse,
    StockInfoResponse,
    BasicInformation,
    TradingInformation,
    KeyMetrics,
    FinancialMetrics,
    StockResponse,
    Interval,
    StockData,
    PredictionResponse,
    StockDatas,
    StockPrice

)
from ..utils.auth import get_current_user

router = APIRouter()
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "ADANIENT.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "TECHM.NS", "BAJAJFINSV.NS", "ONGC.NS",
    "COALINDIA.NS", "M&M.NS", "GRASIM.NS", "TATASTEEL.NS", "ADANIPORTS.NS",
    "DRREDDY.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS", "CIPLA.NS", "BRITANNIA.NS",
    "DIVISLAB.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "UPL.NS", "SBILIFE.NS",
    "LTIM.NS", "BAJAJ-AUTO.NS", "HDFC.NS", "BPCL.NS", "TATACONSUM.NS"
]

NIFTY_IT_STOCKS = [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"
]

NIFTY_BANK_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "AUBANK.NS", "BANDHANBNK.NS"
]

async def get_indian_stock_list():
    """Fetch a list of NSE stocks."""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        return [
            {"symbol": f"{row['SYMBOL']}.NS", "name": row["NAME OF COMPANY"]}
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock list: {str(e)}")


@router.get("/search/{search_text}", response_model=StockSearchResponse)
async def search_stocks(search_text: str, current_user: dict = Depends(get_current_user)):
    """Search for NSE stocks."""
    try:
        companies = await get_indian_stock_list()
        search_str = search_text.upper()
        matching_companies = [
            company
            for company in companies
            if search_str in company["name"].upper() or search_str in company["symbol"].upper()
        ]

        result = []
        for company in matching_companies:
            stock_info = StockInfo(symbol=company["symbol"], name=company["name"])
            try:
                stock = yf.Ticker(company["symbol"])
                info = stock.info
                stock_info.current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )
                stock_info.day_high = info.get("dayHigh")
                stock_info.day_low = info.get("dayLow")
                stock_info.volume = info.get("volume")
            except Exception:
                pass  # Ignore individual stock failures
            result.append(stock_info)

        return StockSearchResponse(companies=result, count=len(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stock/info/{symbol}", response_model=StockInfoResponse)
async def get_stock_info(symbol: str, current_user: dict = Depends(get_current_user)):
    """Fetch detailed stock information."""
    try:
        if not symbol.endswith(".NS"):
            symbol += ".NS"

        stock = yf.Ticker(symbol)
        info = stock.info

        # Handle ex_dividend_date conversion to string
        ex_dividend_date = info.get("exDividendDate")
        if isinstance(ex_dividend_date, (int, float)):
            ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime("%Y-%m-%d")

        # Calculate PEG ratio manually if it's null
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        trailing_eps = info.get("trailingEps")
        forward_eps = info.get("forwardEps")
        
        peg_ratio = None
        if all(v is not None for v in [trailing_eps, forward_eps, pe_ratio]) and trailing_eps > 0:
            # Calculate expected earnings growth rate
            growth_rate = ((forward_eps - trailing_eps) / abs(trailing_eps)) * 100
            if growth_rate > 0:  # Only calculate PEG if growth is positive
                peg_ratio = pe_ratio / growth_rate

        response = StockInfoResponse(
            basic_information=BasicInformation(
                company_name=info.get("longName"),
                symbol=symbol,
                sector=info.get("sector"),
                industry=info.get("industry"),
                website=info.get("website"),
                business_summary=info.get("longBusinessSummary"),
            ),
            trading_information=TradingInformation(
                current_price=info.get("currentPrice"),
                previous_close=info.get("previousClose"),
                open_price=info.get("open"),
                day_low=info.get("dayLow"),
                day_high=info.get("dayHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                volume=info.get("volume"),
                average_volume=info.get("averageVolume"),
                market_cap=info.get("marketCap"),
            ),
            key_metrics=KeyMetrics(
                pe_ratio=pe_ratio,
                forward_pe=forward_pe,
                peg_ratio=peg_ratio,  # Use our calculated PEG ratio
                price_to_book=info.get("priceToBook"),
                trailing_eps=trailing_eps,
                forward_eps=forward_eps,
                book_value=info.get("bookValue"),
                dividend_rate=info.get("dividendRate"),
                dividend_yield=info.get("dividendYield"),
                ex_dividend_date=ex_dividend_date,
            ),
            financial_metrics=FinancialMetrics(
                revenue=info.get("totalRevenue"),
                revenue_per_share=info.get("revenuePerShare"),
                profit_margin=info.get("profitMargins"),
                operating_margin=info.get("operatingMargins"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                total_cash=info.get("totalCash"),
                total_debt=info.get("totalDebt"),
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio"),
            ),
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {symbol}: {str(e)}")

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

@router.get("/api/stock/{symbol}", response_model=StockResponse)
async def get_stock_data(
    symbol: str,
    interval: str = "1d"  # Changed from Interval enum to string
):
    """
    Get historical stock data for the specified symbol and interval.
    
    Parameters:
        symbol: Stock symbol (e.g., 'ITC' or 'ITC.NS')
        interval: Data interval - DAILY (1d), WEEKLY (1wk), or MONTHLY (1mo)
                 Default is '1d' for daily data
    """
    try:
        return fetch_stock_data(symbol, interval)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

class StockManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                await self.disconnect(connection)

stock_manager = StockManager()

# def get_stock_prices(symbol: str) -> dict:
#     """Get price for a single stock with trend analysis"""
#     try:
#         stock = yf.Ticker(symbol)
#         info = stock.info
        
#         # Get current price
#         current_price = None
#         for field in ['regularMarketPrice', 'currentPrice']:
#             if field in info and info[field] is not None:
#                 current_price = info[field]
#                 break
        
#         # Get opening and previous closing prices
#         open_price = info.get('open')
#         prev_close = info.get('previousClose')
        
#         if current_price and (open_price or prev_close):
#             # Calculate price change and trend
#             reference_price = open_price if open_price is not None else prev_close
#             price_change = current_price - reference_price
#             percent_change = (price_change / reference_price) * 100
            
#             return {
#                 'symbol': symbol,
#                 'name': info.get('longName', symbol.replace('.NS', '')),
#                 'price': current_price,
#                 'change': round(price_change, 2),
#                 'change_percent': round(percent_change, 2),
#                 'trend': 'up' if price_change > 0 else 'down' if price_change < 0 else 'neutral',
#                 'volume': info.get('volume', 0),
#                 'market_cap': info.get('marketCap', 0),
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
#         return None
#     except Exception as e:
#         print(f"Error fetching {symbol}: {str(e)}")
#         return None

# async def get_index_stocks(index_stocks: List[str]) -> Dict[str, dict]:
#     """Get prices and trends for all stocks in an index"""
#     prices = {}
#     for symbol in index_stocks:
#         retries = 3
#         for attempt in range(retries):
#             try:
#                 price_data = get_stock_price(symbol)
#                 if price_data:
#                     prices[symbol] = price_data
#                     break
#                 if attempt < retries - 1:
#                     await asyncio.sleep(1)
#             except Exception as e:
#                 print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
#                 if attempt < retries - 1:
#                     await asyncio.sleep(1)
#     return prices

# @router.get("/Get_all_stock")
# async def get_all_stocks():
#     """Get current prices for all tracked stocks"""
#     prices = await get_stock_prices()  # Add await here
#     return JSONResponse(content=prices)

# @router.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     try:
#         await stock_manager.connect(websocket)
#         print(f"Client connected from {websocket.client.host}")
        
#         try:
#             while True:
#                 prices = await get_stock_prices()  # Add await here
#                 if prices:
#                     # Add market summary
#                     advancing = sum(1 for data in prices.values() if data.get('trend') == 'up')
#                     declining = sum(1 for data in prices.values() if data.get('trend') == 'down')
#                     unchanged = sum(1 for data in prices.values() if data.get('trend') == 'neutral')
                    
#                     data_to_send = {
#                         'market_summary': {
#                             'advancing': advancing,
#                             'declining': declining,
#                             'unchanged': unchanged,
#                             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                         },
#                         'stocks': prices
#                     }
#                     await websocket.send_json(data_to_send)
#                 await asyncio.sleep(3)
                
#         except WebSocketDisconnect:
#             print(f"Client disconnected from {websocket.client.host}")
#             stock_manager.disconnect(websocket)
#         except Exception as e:
#             print(f"Error in WebSocket loop: {e}")
#             if websocket in stock_manager.active_connections:
#                 stock_manager.disconnect(websocket)
                
#     except Exception as e:
#         print(f"Error in WebSocket connection: {e}")
#         if websocket in stock_manager.active_connections:
#             stock_manager.disconnect(websocket)


@router.get("/all_indices")
async def get_all_indices():
    """Get data for all indices (Nifty 50, Nifty IT, Nifty Bank) simultaneously"""
    try:
        indices_data = {
            'nifty50': {
                'name': 'NIFTY 50',
                'stocks': await get_index_stocks(NIFTY_50_STOCKS)
            },
            'niftyit': {
                'name': 'NIFTY IT',
                'stocks': await get_index_stocks(NIFTY_IT_STOCKS)
            },
            'niftybank': {
                'name': 'NIFTY BANK',
                'stocks': await get_index_stocks(NIFTY_BANK_STOCKS)
            }
        }

        # Calculate market summary for each index
        for index_data in indices_data.values():
            stocks = index_data['stocks']
            index_data['market_summary'] = {
                'advancing': sum(1 for stock in stocks.values() if stock['trend'] == 'up'),
                'declining': sum(1 for stock in stocks.values() if stock['trend'] == 'down'),
                'unchanged': sum(1 for stock in stocks.values() if stock['trend'] == 'neutral'),
                'total_stocks': len(stocks)
            }

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'indices': indices_data
        }

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching indices data: {str(e)}"
        )

@router.websocket("/ws/all_indices")
async def all_indices_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time updates of all indices"""
    try:
        await stock_manager.connect(websocket)
        print(f"Client connected from {websocket.client.host}")
        
        try:
            while True:
                # Fetch data for all indices
                indices_data = {
                    'nifty50': {
                        'name': 'NIFTY 50',
                        'stocks': await get_index_stocks(NIFTY_50_STOCKS)
                    },
                    'niftyit': {
                        'name': 'NIFTY IT',
                        'stocks': await get_index_stocks(NIFTY_IT_STOCKS)
                    },
                    'niftybank': {
                        'name': 'NIFTY BANK',
                        'stocks': await get_index_stocks(NIFTY_BANK_STOCKS)
                    }
                }

                # Calculate market summary for each index
                for index_data in indices_data.values():
                    stocks = index_data['stocks']
                    index_data['market_summary'] = {
                        'advancing': sum(1 for stock in stocks.values() if stock['trend'] == 'up'),
                        'declining': sum(1 for stock in stocks.values() if stock['trend'] == 'down'),
                        'unchanged': sum(1 for stock in stocks.values() if stock['trend'] == 'neutral'),
                        'total_stocks': len(stocks)
                    }

                # Calculate overall market summary
                total_advancing = sum(index['market_summary']['advancing'] for index in indices_data.values())
                total_declining = sum(index['market_summary']['declining'] for index in indices_data.values())
                total_unchanged = sum(index['market_summary']['unchanged'] for index in indices_data.values())

                data_to_send = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'overall_summary': {
                        'advancing': total_advancing,
                        'declining': total_declining,
                        'unchanged': total_unchanged
                    },
                    'indices': indices_data
                }

                await websocket.send_json(data_to_send)
                await asyncio.sleep(1)  # Update every 3 seconds
                
        except WebSocketDisconnect:
            print(f"Client disconnected from {websocket.client.host}")
            stock_manager.disconnect(websocket)
        except Exception as e:
            print(f"Error in WebSocket loop: {e}")
            if websocket in stock_manager.active_connections:
                stock_manager.disconnect(websocket)
                
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
        if websocket in stock_manager.active_connections:
            stock_manager.disconnect(websocket)

async def get_index_stocks(stock_list: List[str]) -> Dict[str, dict]:
    """Get stock data for a list of symbols with additional metrics"""
    stocks_data = {}
    for symbol in stock_list:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get price information
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            previous_close = info.get('previousClose')
            open_price = info.get('open')
            
            if current_price and previous_close:
                # Calculate changes
                price_change = current_price - previous_close
                percent_change = (price_change / previous_close) * 100
                
                stocks_data[symbol] = {
                    'symbol': symbol,
                    'name': info.get('longName', symbol.replace('.NS', '')),
                    'price': current_price,
                    'change': round(price_change, 2),
                    'change_percent': round(percent_change, 2),
                    'trend': 'up' if price_change > 0 else 'down' if price_change < 0 else 'neutral',
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'open': open_price,
                    'prev_close': previous_close,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
            
    return stocks_data