"""
Kraken data provider for institutional-grade crypto analysis.
"""
import ccxt
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from app.providers.base import BaseProvider, exponential_backoff

logger = logging.getLogger(__name__)


class KrakenProvider(BaseProvider):
    """Kraken crypto data provider using ccxt for institutional analysis."""
    
    def __init__(self):
        super().__init__("kraken")
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize Kraken exchange with error handling."""
        try:
            self.exchange = ccxt.kraken({
                'sandbox': False,
                'rateLimit': 1000,  # Kraken rate limit
                'enableRateLimit': True,
            })
            self.logger.info("✅ Kraken exchange initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kraken exchange: {e}")
    
    @exponential_backoff(max_retries=3)
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for a symbol from Kraken."""
        try:
            if not self.exchange:
                self._initialize_exchange()
                
            if not self.exchange:
                return None
                
            ticker = self.exchange.fetch_ticker(symbol)
            # Convert ticker object to dict
            return dict(ticker) if ticker else None
            
        except Exception as e:
            self.logger.debug(f"Failed to get ticker for {symbol}: {e}")
            return None
    
    @exponential_backoff(max_retries=3)
    def get_klines(self, symbol: str, interval: str = '5m', limit: int = 500) -> Optional[List]:
        """Get kline/candlestick data from Kraken."""
        try:
            if not self.exchange:
                self._initialize_exchange()
                
            if not self.exchange:
                return None
            
            # Kraken timeframe mapping
            timeframe_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            kraken_interval = timeframe_map.get(interval, '5m')
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, kraken_interval, limit=limit)
            
            # Convert to expected format
            klines = []
            for candle in ohlcv:
                klines.append([
                    candle[0],  # timestamp
                    candle[1],  # open
                    candle[2],  # high
                    candle[3],  # low
                    candle[4],  # close
                    candle[5],  # volume
                ])
            
            return klines
            
        except Exception as e:
            self.logger.debug(f"Failed to get klines for {symbol}: {e}")
            return None
    
    def get_top_symbols(self, limit: int = 100) -> List[str]:
        """Get top trading symbols on Kraken by volume."""
        try:
            if not self.exchange:
                self._initialize_exchange()
                
            if not self.exchange:
                return []
            
            # Get all markets
            markets = self.exchange.load_markets()
            
            # Get tickers for volume data
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USD pairs and active markets
            usd_pairs = []
            for symbol, market in markets.items():
                if (market['active'] and 
                    market['quote'] in ['USD', 'USDT', 'USDC'] and
                    symbol in tickers):
                    
                    ticker = tickers[symbol]
                    volume_24h = ticker.get('quoteVolume', 0)
                    
                    # Convert to float for comparison
                    try:
                        volume_float = float(volume_24h) if volume_24h else 0
                        if volume_float > 100000:  # Minimum $100k volume
                            usd_pairs.append({
                                'symbol': symbol,
                                'volume': volume_float
                            })
                    except (ValueError, TypeError):
                        continue
            
            # Sort by volume and return top symbols
            usd_pairs.sort(key=lambda x: x['volume'], reverse=True)
            
            top_symbols = [pair['symbol'] for pair in usd_pairs[:limit]]
            
            self.logger.info(f"Found {len(top_symbols)} active Kraken symbols")
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get Kraken symbols: {e}")
            # Return some common Kraken pairs as fallback
            return [
                'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
                'XRP/USD', 'LTC/USD', 'BCH/USD', 'ATOM/USD', 'ALGO/USD',
                'MATIC/USD', 'SOL/USD', 'AVAX/USD', 'LUNA/USD', 'FIL/USD'
            ]
    
    def get_24h_stats(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get 24h statistics for a symbol."""
        try:
            ticker = self.get_ticker(symbol)
            if not ticker:
                return None
                
            return {
                'price_change_24h': ticker.get('change', 0),
                'price_change_percent_24h': ticker.get('percentage', 0),
                'volume_24h': ticker.get('quoteVolume', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0),
                'last_price': ticker.get('last', 0)
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to get 24h stats for {symbol}: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if Kraken market is open (crypto markets are always open)."""
        return True
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information."""
        try:
            if not self.exchange:
                self._initialize_exchange()
                
            return {
                'name': 'Kraken',
                'id': 'kraken',
                'status': 'operational' if self.exchange else 'error',
                'rate_limit': 1000,
                'has_spot': True,
                'has_futures': True,
                'countries': ['US'],
                'urls': {
                    'api': 'https://api.kraken.com',
                    'www': 'https://www.kraken.com',
                    'doc': 'https://docs.kraken.com/rest/'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            return {'name': 'Kraken', 'status': 'error'}


class KrakenMultiProvider:
    """Multi-provider wrapper for Kraken with enhanced features."""
    
    def __init__(self):
        self.kraken = KrakenProvider()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_institutional_symbols(self, min_volume: float = 1000000) -> List[str]:
        """Get institutional-grade symbols with high volume."""
        try:
            all_symbols = self.kraken.get_top_symbols(200)
            institutional_symbols = []
            
            for symbol in all_symbols:
                stats = self.kraken.get_24h_stats(symbol)
                if stats and stats.get('volume_24h', 0) >= min_volume:
                    institutional_symbols.append(symbol)
                    
                if len(institutional_symbols) >= 50:  # Limit for performance
                    break
            
            self.logger.info(f"Found {len(institutional_symbols)} institutional-grade Kraken symbols")
            return institutional_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get institutional symbols: {e}")
            return []
    
    def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data for institutional analysis."""
        try:
            ticker = self.kraken.get_ticker(symbol)
            klines_5m = self.kraken.get_klines(symbol, '5m', 100)
            klines_1h = self.kraken.get_klines(symbol, '1h', 100)
            klines_1d = self.kraken.get_klines(symbol, '1d', 100)
            stats_24h = self.kraken.get_24h_stats(symbol)
            
            return {
                'symbol': symbol,
                'ticker': ticker,
                'klines_5m': klines_5m,
                'klines_1h': klines_1h, 
                'klines_1d': klines_1d,
                'stats_24h': stats_24h,
                'exchange': 'kraken',
                'institutional_grade': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive data for {symbol}: {e}")
            return {}
