"""
Base provider functionality with retries, exponential backoff, and utilities.
"""
import time
import logging
import pandas as pd
from typing import Optional, Callable, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

logger = logging.getLogger(__name__)


def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for exponential backoff retry logic."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def parallel_map(fn: Callable, items: List[Any], max_workers: int = 16, timeout: int = 20) -> List[Any]:
    """Execute function in parallel over items with timeout and error handling."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(fn, item): item for item in items}
        
        for future in as_completed(future_to_item, timeout=timeout):
            item = future_to_item[future]
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                logger.debug(f"parallel_map failed for {item}: {e}")
                results.append(None)  # Add None for failed items to maintain order
                
    return results


def parallel_fetch(fetch_func: Callable, symbols: List[str], max_workers: int = 5) -> Dict[str, Any]:
    """Execute fetch function in parallel for multiple symbols."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_func, symbol): symbol for symbol in symbols}
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                
    return results


def last_bar_age_minutes(df: pd.DataFrame) -> Optional[float]:
    """Calculate age of last bar in minutes from current time."""
    if df is None or df.empty:
        return None
    
    try:
        last_timestamp = df.index[-1]
        if pd.isna(last_timestamp):
            return None
            
        # Ensure timezone awareness
        if last_timestamp.tz is None:
            last_timestamp = pd.Timestamp(last_timestamp, tz='UTC')
        
        current_time = pd.Timestamp.now(tz='UTC')
        age_delta = current_time - last_timestamp
        return age_delta.total_seconds() / 60.0
        
    except Exception as e:
        logger.error(f"Error calculating last bar age: {e}")
        return None


def stitch_last_trade(df: pd.DataFrame, last_price: float, last_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Stitch last trade price to the most recent bar in the DataFrame.
    
    Args:
        df: OHLCV DataFrame with datetime index
        last_price: Most recent trade price
        last_timestamp: Optional timestamp of last trade (defaults to now)
    
    Returns:
        DataFrame with updated last bar or new bar appended
    """
    if df is None or df.empty or last_price is None:
        return df
    
    try:
        df = df.copy()
        
        if last_timestamp is None:
            last_timestamp = pd.Timestamp.now(tz='UTC')
        elif last_timestamp.tz is None:
            last_timestamp = pd.Timestamp(last_timestamp, tz='UTC')
        
        last_bar_time = df.index[-1]
        if last_bar_time.tz is None:
            last_bar_time = pd.Timestamp(last_bar_time, tz='UTC')
        
        # If the last trade is within the same timeframe as the last bar, update it
        time_diff = (last_timestamp - last_bar_time).total_seconds() / 60.0
        
        # Determine timeframe from the DataFrame (approximate)
        if len(df) > 1:
            tf_minutes = (df.index[-1] - df.index[-2]).total_seconds() / 60.0
        else:
            tf_minutes = 60  # Default to 1 hour
        
        if time_diff < tf_minutes:
            # Update the last bar's close and potentially high/low using safe column access
            try:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                high_col = 'High' if 'High' in df.columns else 'high'
                low_col = 'Low' if 'Low' in df.columns else 'low'
                
                # Update close price
                df.at[df.index[-1], close_col] = last_price
                
                # Update high if last_price is higher
                current_high = df.at[df.index[-1], high_col]
                if last_price > current_high:
                    df.at[df.index[-1], high_col] = last_price
                    
                # Update low if last_price is lower
                current_low = df.at[df.index[-1], low_col]
                if last_price < current_low:
                    df.at[df.index[-1], low_col] = last_price
            except Exception as inner_e:
                logger.error(f"Error updating last bar: {inner_e}")
        else:
            # Create a new bar if the timestamp is significantly newer
            try:
                # Use proper column names
                open_col = 'Open' if 'Open' in df.columns else 'open'
                high_col = 'High' if 'High' in df.columns else 'high'
                low_col = 'Low' if 'Low' in df.columns else 'low'
                close_col = 'Close' if 'Close' in df.columns else 'close'
                volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
                
                new_bar = pd.DataFrame({
                    open_col: [last_price],
                    high_col: [last_price], 
                    low_col: [last_price],
                    close_col: [last_price],
                    volume_col: [0]  # Unknown volume for live price
                }, index=[last_timestamp])
                
                # Ensure column order matches
                new_bar = new_bar[df.columns]
                df = pd.concat([df, new_bar])
            except Exception as inner_e:
                logger.error(f"Error creating new bar: {inner_e}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error stitching last trade: {e}")
        return df


class BaseProvider:
    """Base class for data providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data. To be implemented by subclasses."""
        raise NotImplementedError
    
    def fetch_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch last trade price and timestamp. To be implemented by subclasses."""
        raise NotImplementedError
    
    def get_symbols(self, quote_currency: str = None, limit: int = 50) -> List[str]:
        """Get available symbols. To be implemented by subclasses."""
        raise NotImplementedError
