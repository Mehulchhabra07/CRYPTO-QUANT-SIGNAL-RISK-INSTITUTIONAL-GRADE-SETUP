"""
Crypto data providers with Binance primary and Coinbase fallback.
"""
import ccxt
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from app.providers.base import BaseProvider, exponential_backoff

logger = logging.getLogger(__name__)


class BinanceProvider(BaseProvider):
    """Binance crypto data provider using ccxt."""
    
    def __init__(self):
        super().__init__("binance")
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize Binance exchange with error handling."""
        try:
            self.exchange = ccxt.binance({
                'sandbox': False,
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance exchange: {e}")
    
    @exponential_backoff(max_retries=3)
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance."""
        if not self.exchange:
            return None
            
        try:
            # Normalize timeframe for ccxt
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            ccxt_tf = tf_map.get(timeframe, timeframe)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, ccxt_tf, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise
    
    @exponential_backoff(max_retries=2)
    def fetch_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch last trade price from Binance."""
        if not self.exchange:
            return None
            
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'price': ticker['last'],
                'timestamp': pd.Timestamp(ticker['timestamp'], unit='ms', tz='UTC') if ticker['timestamp'] else pd.Timestamp.now(tz='UTC'),
                'volume': ticker.get('baseVolume', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch last trade for {symbol}: {e}")
            raise
    
    def get_symbols(self, quote_currency: str = "USDT", limit: int = 50) -> List[str]:
        """Get top symbols by volume with better filtering."""
        if not self.exchange:
            return []
            
        try:
            markets = self.exchange.load_markets()
            
            # Filter for valid, active markets with sufficient volume
            valid_symbols = []
            for symbol, market in markets.items():
                if (market['quote'] == quote_currency and 
                    market['active'] and 
                    market.get('spot', True) and  # Only spot markets
                    not symbol.endswith('BULL/USDT') and  # Exclude leveraged tokens
                    not symbol.endswith('BEAR/USDT') and
                    not symbol.endswith('UP/USDT') and
                    not symbol.endswith('DOWN/USDT') and
                    '/' in symbol):  # Ensure proper format
                    valid_symbols.append(symbol)
            
            # Get volume data to sort by popularity
            max_fetch = min(len(valid_symbols), max(limit * 3, 400))  # Fetch more to ensure we get liquid ones
            
            if not valid_symbols:
                return []
                
            try:
                tickers = self.exchange.fetch_tickers(valid_symbols[:max_fetch])
            except Exception as e:
                self.logger.warning(f"Failed to fetch tickers, using subset: {e}")
                # Try with smaller batch if it fails
                tickers = self.exchange.fetch_tickers(valid_symbols[:100])
            
            # Sort by volume and filter by minimum volume threshold
            symbol_volumes = []
            for symbol, ticker in tickers.items():
                volume = ticker.get('quoteVolume', 0)
                try:
                    volume_float = float(volume) if volume is not None else 0.0
                    if volume_float > 100000:  # Minimum $100k daily volume
                        symbol_volumes.append((symbol, volume_float))
                except (ValueError, TypeError):
                    continue
            
            if not symbol_volumes:
                # Fallback to any volume if no symbols meet threshold
                for symbol, ticker in tickers.items():
                    volume = ticker.get('quoteVolume', 0)
                    try:
                        volume_float = float(volume) if volume is not None else 0.0
                        symbol_volumes.append((symbol, volume_float))
                    except (ValueError, TypeError):
                        symbol_volumes.append((symbol, 0.0))
            
            sorted_symbols = sorted(symbol_volumes, key=lambda x: x[1], reverse=True)
            
            result = [symbol for symbol, _ in sorted_symbols[:limit]]
            self.logger.info(f"Found {len(result)} valid {quote_currency} symbols with good volume")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []


class CoinbaseProvider(BaseProvider):
    """Coinbase Pro fallback provider."""
    
    def __init__(self):
        super().__init__("coinbase")
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize Coinbase exchange."""
        try:
            self.exchange = ccxt.coinbase({
                'sandbox': False,
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize Coinbase exchange: {e}")
    
    @exponential_backoff(max_retries=3)
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Coinbase."""
        if not self.exchange:
            return None
            
        try:
            # Convert Binance symbol format to Coinbase
            symbol = symbol.replace('USDT', 'USD').replace('/', '-')
            
            # Timeframe mapping
            tf_map = {
                '1m': '60', '5m': '300', '15m': '900', '30m': '1800',
                '1h': '3600', '4h': '14400', '1d': '86400'
            }
            ccxt_tf = tf_map.get(timeframe, '3600')
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, ccxt_tf, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise
    
    @exponential_backoff(max_retries=2)
    def fetch_last_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch last trade from Coinbase."""
        if not self.exchange:
            return None
            
        try:
            symbol = symbol.replace('USDT', 'USD').replace('/', '-')
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'price': ticker['last'],
                'timestamp': pd.Timestamp(ticker['timestamp'], unit='ms', tz='UTC') if ticker['timestamp'] else pd.Timestamp.now(tz='UTC'),
                'volume': ticker.get('baseVolume', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch last trade for {symbol}: {e}")
            raise
    
    def get_symbols(self, quote_currency: str = "USD", limit: int = 50) -> List[str]:
        """Get symbols from Coinbase."""
        if not self.exchange:
            return []
            
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol, market in markets.items() 
                      if market['quote'] == quote_currency and market['active']]
            return symbols[:limit]
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []


class CryptoDataManager:
    """Manages crypto data fetching using enhanced Binance provider with proven performers."""
    
    def __init__(self):
        self.primary = BinanceProvider()
        self.logger = logging.getLogger(f"{__name__}.CryptoDataManager")
        self._valid_symbols_cache = None
        self._cache_time = None
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for trading."""
        try:
            # Basic format validation
            if '/' not in symbol:
                return False
            
            base, quote = symbol.split('/')
            
            # Must be reasonable length
            if len(base) < 2 or len(base) > 10 or len(quote) < 3 or len(quote) > 5:
                return False
            
            # Check against common quote currencies
            valid_quotes = ['USDT', 'USD', 'BTC', 'ETH', 'BUSD']
            if quote not in valid_quotes:
                return False
            
            # Exclude known problematic patterns
            excluded_patterns = ['BULL', 'BEAR', 'UP', 'DOWN', '3L', '3S']
            for pattern in excluded_patterns:
                if pattern in base:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_and_filter_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and filter symbol list."""
        valid_symbols = []
        for symbol in symbols:
            if self._is_valid_symbol(symbol):
                valid_symbols.append(symbol)
            else:
                self.logger.debug(f"Filtered out invalid symbol: {symbol}")
        
        return valid_symbols
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV with symbol validation."""
        # Validate symbol before attempting fetch
        if not self._is_valid_symbol(symbol):
            self.logger.warning(f"Invalid symbol format: {symbol}")
            return None
            
        try:
            result = self.primary.fetch_ohlcv(symbol, timeframe, limit)
            if result is not None and not result.empty:
                return result
        except Exception as e:
            self.logger.warning(f"Binance provider failed for {symbol}: {e}")
        
        self.logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    def fetch_multi_timeframe(self, symbol: str, tf_lower: str, tf_higher: str, 
                            limit_lower: int = 1200, limit_higher: int = 600) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch multiple timeframes with validation and last trade stitching."""
        # Validate symbol before attempting fetch
        if not self._is_valid_symbol(symbol):
            self.logger.warning(f"Invalid symbol format: {symbol}")
            return None
            
        try:
            # Fetch both timeframes
            df_lower = self.fetch_ohlcv(symbol, tf_lower, limit_lower)
            df_higher = self.fetch_ohlcv(symbol, tf_higher, limit_higher)
            
            if df_lower is None and df_higher is None:
                return None
            
            # Try to get last trade for freshness
            last_trade = None
            try:
                last_trade = self.primary.fetch_last_trade(symbol)
            except Exception:
                pass
            
            # Stitch last trade if available
            if last_trade and last_trade.get('price'):
                from app.providers.base import stitch_last_trade
                if df_lower is not None:
                    df_lower = stitch_last_trade(df_lower, last_trade['price'], last_trade.get('timestamp'))
                if df_higher is not None:
                    df_higher = stitch_last_trade(df_higher, last_trade['price'], last_trade.get('timestamp'))
            
            result = {}
            if df_lower is not None:
                result[tf_lower] = df_lower
            if df_higher is not None:
                result[tf_higher] = df_higher
                
            return result if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to fetch multi-timeframe data for {symbol}: {e}")
            return None
    
    def get_symbols(self, quote_currency: str = "USDT", limit: int = 50) -> List[str]:
        """Get symbols using enhanced Binance filtering with proven performers."""
        try:
            # First try to get symbols from Binance with better filtering
            raw_symbols = self.primary.get_symbols(quote_currency, limit * 2)  # Get more to filter
            
            if raw_symbols:
                # Validate and filter symbols
                valid_symbols = self._validate_and_filter_symbols(raw_symbols)
                
                # Add proven top performers to the list if they're not already included
                top_performers = [
                    'BTC/USDT', 'ETH/USDT', 'COW/USDT', 'SOL/USDT', 'TON/USDT',
                    'USDC/USDT', 'TRX/USDT', 'AAVE/USDT', 'CRV/USDT', 'PEPE/USDT',
                    'SEI/USDT', 'SHIB/USDT', 'XRP/USDT', 'ADA/USDT', 'BNB/USDT'
                ]
                
                # Combine and dedupe, prioritizing proven performers
                final_symbols = []
                added_symbols = set()
                
                # Add top performers first (if valid)
                for symbol in top_performers:
                    if (symbol not in added_symbols and 
                        self._is_valid_symbol(symbol) and 
                        len(final_symbols) < limit):
                        final_symbols.append(symbol)
                        added_symbols.add(symbol)
                
                # Add validated Binance results to fill the rest
                for symbol in valid_symbols:
                    if symbol not in added_symbols and len(final_symbols) < limit:
                        final_symbols.append(symbol)
                        added_symbols.add(symbol)
                
                self.logger.info(f"Returning {len(final_symbols)} validated symbols including proven performers")
                return final_symbols
                
        except Exception as e:
            self.logger.warning(f"Enhanced symbol fetching failed: {e}")
        
        # Enhanced fallback with more proven symbols
        fallback_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'TRX/USDT', 'TON/USDT', 'AVAX/USDT',
            'SHIB/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
            'LTC/USDT', 'BCH/USDT', 'NEAR/USDT', 'ATOM/USDT', 'AAVE/USDT'
        ]
        
        # Validate fallback symbols too
        validated_fallback = self._validate_and_filter_symbols(fallback_symbols)
        
        self.logger.info(f"Using enhanced fallback symbols: {len(validated_fallback[:limit])}")
        return validated_fallback[:limit]
