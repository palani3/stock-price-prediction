from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Optional, Union
from enum import Enum

# First, define the StockData model that will be used by StockResponse
class StockData(BaseModel):
    date: str
    open: float
    close: float
    high: float
    low: float
    volume: int

class StockResponse(BaseModel):
    symbol: str
    interval: str
    data: List[StockData]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_records: int
# Define the Interval enum for valid time intervals
class Interval(str, Enum):
    """
    Enumeration for the three main trading intervals we want to support.
    These intervals are most commonly used for technical analysis and long-term trading.
    """
    DAILY = "1d"      # Daily data points
    WEEKLY = "1wk"    # Weekly aggregated data
    MONTHLY = "1mo"   # Monthly aggregated data

# Other models for stock information
class StockInfo(BaseModel):
    """
    Basic stock information model.
    Used for quick lookups and search results.
    """
    symbol: str
    name: str
    current_price: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None

class StockSearchResponse(BaseModel):
    """
    Container for stock search results.
    """
    companies: List[StockInfo]
    count: int

class BasicInformation(BaseModel):
    company_name: Optional[str]
    symbol: str
    sector: Optional[str]
    industry: Optional[str]
    website: Optional[str]
    business_summary: Optional[str]

class TradingInformation(BaseModel):
    current_price: Optional[float]
    previous_close: Optional[float]
    open_price: Optional[float]
    day_low: Optional[float]
    day_high: Optional[float]
    fifty_two_week_low: Optional[float]
    fifty_two_week_high: Optional[float]
    volume: Optional[int]
    average_volume: Optional[int]
    market_cap: Optional[int]

class KeyMetrics(BaseModel):
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    trailing_eps: Optional[float]
    forward_eps: Optional[float]
    book_value: Optional[float]
    dividend_rate: Optional[float]
    dividend_yield: Optional[float]
    ex_dividend_date: Optional[str]

class FinancialMetrics(BaseModel):
    revenue: Optional[float]
    revenue_per_share: Optional[float]
    profit_margin: Optional[float]
    operating_margin: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    total_cash: Optional[float]
    total_debt: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]

class StockInfoResponse(BaseModel):
    basic_information: BasicInformation
    trading_information: TradingInformation
    key_metrics: KeyMetrics
    financial_metrics: FinancialMetrics
























class StockDatas(BaseModel):
    symbol: str
    interval: str
    data: List[Dict[str, Any]]
    start_date: str
    end_date: str
    total_records: int

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    support_level: float
    resistance_level: float
    model_accuracy: float
    recent_signals: List[Dict[str, str]]
    technical_indicators: Dict[str, float]













class StockPrice(BaseModel):
    price: float
    timestamp: str




class StockDatas(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class StockResponse(BaseModel):
    symbol: str
    interval: str
    data: List[StockDatas]
    start_date: Optional[str]
    end_date: Optional[str]
    total_records: int

class TrainingResponse(BaseModel):
    model_accuracy: float
    validation_accuracy: float
    latest_prediction: str
    prediction_confidence: float
    support_level: float
    resistance_level: float
    trading_signals: Dict[str, str]

    class Config:
        json_schema_extra = {
            "example": {
                "model_accuracy": 0.75,
                "validation_accuracy": 0.72,
                "latest_prediction": "buy",
                "prediction_confidence": 0.85,
                "support_level": 1200.50,
                "resistance_level": 1300.75,
                "trading_signals": {
                    "2024-01-01": "buy",
                    "2024-01-02": "hold",
                    "2024-01-03": "sell"
                }
            }
        }


