import pandas as pd
import numpy as np
import math
from datetime import datetime, time
import pickle
import os
import hashlib
from numba import jit

@jit(nopython=True)
def find_exit_buy_numba(headers, lowers, closes, entry_idx, target_price, stop_price):
    n = len(headers)
    # Start looking from the NEXT candle (entry is at close of entry_idx-1, so next bar is entry_idx)
    # Or is entry_idx the index of the signal candle?
    # In simulate_buy_trade: entry_price = close[entry_idx - 1].
    # Loop starts at 'range(entry_idx, len(df))'. entry_idx is "i" from the loop, which is the signal candle index?
    # Wait, check logic: 
    # run_backtest calls simulate_trade(i, ...). "i" is the current candle being checked.
    # If signal triggers at "i", we enter at Close[i]. The trade starts evolving at i+1.
    # The original loop was `for j in range(entry_idx, len(self.df))` where entry_idx was passed as 'i'.
    # BUT wait, simulate_buy_trade receives 'entry_idx'.
    # Let's check 'run_backtest'. It iterates `for i in range(20, len(self.df)):`
    # and calls `self.simulate_buy_trade(i, ...)`
    # Inside simulate: entry_price = values[entry_idx - 1] ??? 
    # Wait, if I call with 'i', entry_price should be Close[i].
    # Why [entry_idx - 1]? 
    # Ah, let's re-read simulate_buy_trade carefully.
    
    # Line 659: entry_price = self.df['close'].values[entry_idx - 1]
    # This implies 'entry_idx' passed is the index where the trade IS ALREADY HAPPENING?
    # Or is 'entry_idx' the index current loop?
    
    # If checking logic check_buy_conditions(i): returns true based on [i-1], [i-2]...
    # The signal is confirmed at Close of 'i'.
    # We enter at Close of 'i'.
    # The trade results come from 'i+1' onwards?
    
    # Original loop: `for j in range(entry_idx, len(self.df))`
    # If entry_idx is 'i', then we check High[i] and Low[i].
    # But if we enter at Close[i], we cannot hit TP/SL at candle 'i' (unless intra-bar, but we use Close).
    # Assuming standard backtest: Signal at Close[i], Open trade. Check limits starting i+1.
    # But existing code does `for j in range(entry_idx, ...)`. 
    # If entry_idx is 'i', it checks current bar 'i'.
    # If entry is Close[i-1] (as in line 659), then checking 'i' is correct.
    # So 'entry_idx' MUST be the index of the first candle AFTER the signal candle.
    # Let's keep the exact logic of the original code to avoid breaking behavior.
    # Original: entry_price = close[entry_idx-1]. Loop starts at entry_idx.
    
    for j in range(entry_idx, n):
        if headers[j] >= target_price:
            return target_price, True, j - entry_idx + 1
        if lowers[j] <= stop_price:
            return stop_price, False, j - entry_idx + 1
            
    return closes[n-1], False, n - entry_idx

@jit(nopython=True)
def find_exit_sell_numba(headers, lowers, closes, entry_idx, target_price, stop_price):
    n = len(headers)
    for j in range(entry_idx, n):
        if lowers[j] <= target_price:
            return target_price, True, j - entry_idx + 1
        if headers[j] >= stop_price:
            return stop_price, False, j - entry_idx + 1
            
    return closes[n-1], False, n - entry_idx

@jit(nopython=True)
def simulate_trades_batch_numba(entry_indices, directions, opens, highs, lows, closes, ranges, tp_mult, sl_mult):
    n_trades = len(entry_indices)
    n_candles = len(closes)
    
    results = np.zeros(n_trades, dtype=np.float64)
    winners = np.zeros(n_trades, dtype=np.bool_)
    exit_prices = np.zeros(n_trades, dtype=np.float64)
    exit_indices = np.zeros(n_trades, dtype=np.int64)
    exit_bars = np.zeros(n_trades, dtype=np.int64)
    targets = np.zeros(n_trades, dtype=np.float64)
    stops = np.zeros(n_trades, dtype=np.float64)
    
    for k in range(n_trades):
        idx = entry_indices[k]
        direction = directions[k] # 1 for Buy, -1 for Sell
        crs = ranges[k]
        entry_price = closes[idx-1]
        
        target_price = 0.0
        stop_price = 0.0
        
        if direction == 1: # Buy
            target_price = entry_price + (crs * tp_mult)
            stop_price = entry_price - (crs * sl_mult)
        else: # Sell
            target_price = entry_price - (crs * tp_mult)
            stop_price = entry_price + (crs * sl_mult)
            
        targets[k] = target_price
        stops[k] = stop_price
            
        # Search Exit
        # Logic copied from find_exit_buy_numba/sell
        # To avoid function call overhead inside loop if possible, or just call them (numba inlines usually)
        # Putting loop here for clarity
        
        ep = closes[n_candles-1] # Default exit at end
        win = False
        bars = 0
        found = False
        
        for j in range(idx, n_candles):
            if direction == 1: # Buy
                if highs[j] >= target_price:
                    ep = target_price
                    win = True
                    bars = j - idx + 1
                    found = True
                    break
                if lows[j] <= stop_price:
                    ep = stop_price
                    win = False
                    bars = j - idx + 1
                    found = True
                    break
            else: # Sell
                if lows[j] <= target_price:
                    ep = target_price
                    win = True
                    bars = j - idx + 1
                    found = True
                    break
                if highs[j] >= stop_price:
                    ep = stop_price
                    win = False
                    bars = j - idx + 1
                    found = True
                    break
        
        if not found:
            bars = n_candles - idx
            
        exit_prices[k] = ep
        winners[k] = win
        exit_indices[k] = idx + bars - 1 # approximate
        exit_bars[k] = bars
        
        if direction == 1:
            results[k] = ep - entry_price
        else:
            results[k] = entry_price - ep
            
    return results, winners, exit_prices, exit_bars, targets, stops


class StrategyAnalyzer:
    # Proporções disponíveis para Take x Stop
    AVAILABLE_RATIOS = [
        {'tp': 1, 'sl': 4, 'label': 'TP 1x : SL 4x'},
        {'tp': 1, 'sl': 3, 'label': 'TP 1x : SL 3x'},
        {'tp': 1, 'sl': 2, 'label': 'TP 1x : SL 2x'},
        {'tp': 1, 'sl': 1, 'label': 'TP 1x : SL 1x'},
        {'tp': 2, 'sl': 1, 'label': 'TP 2x : SL 1x'},
        {'tp': 3, 'sl': 1, 'label': 'TP 3x : SL 1x'},
        {'tp': 4, 'sl': 1, 'label': 'TP 4x : SL 1x'},
    ]
    
    def __init__(self, data_file, target_points=50, stop_points=50, strategy_type='inside_bar', data_frame=None):
        self.data_file = data_file
        self.target_points = target_points
        self.stop_points = stop_points
        self.strategy_type = strategy_type
        self.df = data_frame
        self.signals = []
        self.all_trades_df = pd.DataFrame() 
        self.current_ratio = '1:2'
        self.trades = []
        self.trades_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self._filtered_df = None
        
    def _get_cache_filename(self):
        """Gera um nome de arquivo único para o cache baseado no arquivo de dados e na estratégia"""
        if not os.path.exists('.cache'):
            os.makedirs('.cache', exist_ok=True)

        
        if not os.path.exists('.cache'):
            os.makedirs('.cache', exist_ok=True)
            
        # Optimization: Use file modification time and size instead of reading content hash
        # This is orders of magnitude faster for large files (like 500MB+)
        stat = os.stat(self.data_file)
        file_hash = f"{stat.st_mtime}_{stat.st_size}"
        
        # Use simple hash of this string
        file_hash = hashlib.md5(file_hash.encode()).hexdigest()
        
        return f".cache/strategy_{self.strategy_type}_{file_hash}_opt_v3_acc.parquet"

    def load_cache(self):
        """Tenta carregar resultados do cache (Parquet)"""
        cache_file = self._get_cache_filename()
        if os.path.exists(cache_file):
            try:
                print(f"Loading cache from {cache_file}...")
                self.all_trades_df = pd.read_parquet(cache_file)
                return not self.all_trades_df.empty
            except Exception as e:
                print(f"Error loading cache: {e}")
                return False
        return False
        return False

    def save_cache(self):
        """Salva resultados no cache (Parquet)"""
        try:
            cache_file = self._get_cache_filename()
            print(f"Saving cache to {cache_file}...")
            # Ensure specialized types are handled if necessary, but basic types work well in parquet
            self.all_trades_df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_data(self):
        # Optimization: If DF was passed in constructor, skip loading
        if self.df is not None:
            # Still valid to calculate derived columns if needed?
            # Assuming shared DF is already fully processed.
            return
            
        # Determine paths
        is_parquet = self.data_file.endswith('.parquet')
        
        if is_parquet:
            print(f"Loading data from Parquet: {self.data_file}")
            self.df = pd.read_parquet(self.data_file)
        else:
            parquet_path = self.data_file.replace('.txt', '.parquet').replace('.csv', '.parquet')
            if os.path.exists(parquet_path):
                print(f"Loaded from Parquet (auto-detected): {parquet_path}")
                self.df = pd.read_parquet(parquet_path)
            else:
                print(f"Loading from Text/CSV: {self.data_file}")
                # For our specific WIN file, it has no headers if it's the 1.6M rows one, 
                # but pandas usually detects them if they exist.
                # Let's try to detect if it's our comma-separated full file.
                try:
                    self.df = pd.read_csv(self.data_file)
                except:
                    # Fallback for tab-separated or other formats if needed
                    self.df = pd.read_csv(self.data_file, sep=None, engine='python')
                
                self.df['time'] = pd.to_datetime(self.df['time'])
                # Save as Parquet for future fast loading
                try:
                    self.df.to_parquet(parquet_path, index=False)
                    print(f"Converted and cached to Parquet: {parquet_path}")
                except Exception as e:
                    print(f"Could not save Parquet: {e}")
        
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df['hour'] = self.df['time'].dt.hour
        self.df['weekday'] = self.df['time'].dt.dayofweek
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['year_month'] = self.df['time'].dt.to_period('M').astype(str)
        self.df['body_high'] = self.df[['open', 'close']].max(axis=1)
        self.df['body_low'] = self.df[['open', 'close']].min(axis=1)
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['sma20'] = self.df['close'].rolling(window=20).mean()
        
        # Calculate ATR for angle normalization
        self.df['prev_close'] = self.df['close'].shift(1)
        self.df['tr'] = np.maximum(self.df['high'] - self.df['low'],
                           np.maximum((self.df['high'] - self.df['prev_close']).abs(),
                                      (self.df['low'] - self.df['prev_close']).abs()))
        self.df['atr'] = self.df['tr'].rolling(window=20).mean()

        # 3. Volatility
        self.df['std_dev'] = self.df['close'].rolling(window=20).std()

        # 6. STD Reversal Strategy
        std_candle = self.df[['high', 'low', 'close']].std(axis=1)
        
        # Sell Condition (Topo): STD > (Close - Low) * 1.2
        # Detects Shooting Star (Close near Low -> Small diff -> Easy trigger)
        cl_diff = self.df['close'] - self.df['low']
        sell_cond = std_candle > (cl_diff * 1.2)
        self.df['rev_std_sell_signal'] = sell_cond.rolling(window=4).sum()
        
        # Buy Condition (Fundo): STD > (High - Close) * 1.2
        # Detects Hammer (Close near High -> Small diff -> Easy trigger)
        hc_diff = self.df['high'] - self.df['close']
        buy_cond = std_candle > (hc_diff * 1.2)
        self.df['rev_std_buy_signal'] = buy_cond.rolling(window=4).sum()
        
        # 4. Dist SMA
        self.df['dist_sma'] = (self.df['close'] - self.df['sma20']) / (self.df['sma20'] + 1e-9) * 1000.0
        
        # 5. Body/Wicks
        range_c = self.df['high'] - self.df['low']
        self.df['body_pct'] = (self.df['close'] - self.df['open']).abs() / (range_c + 1e-9)
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['wick_u_pct'] = self.df['upper_wick'] / (range_c + 1e-9)
        self.df['wick_l_pct'] = self.df['lower_wick'] / (range_c + 1e-9)
        
        # 7. Distance Index (Previous Day's Most Important Price) - VECTORIZED
        self.df['date'] = self.df['time'].dt.date
        self.df['close_rounded'] = self.df['close'].round(-1)
        
        # Vectorized: Calculate importance per price level per day
        importance_df = (self.df.groupby(['date', 'close_rounded'])
            .agg(count=('close', 'size'), vol_sum=('tick_volume', 'sum'))
            .reset_index())
        importance_df['importance'] = importance_df['count'] * importance_df['vol_sum']
        
        # Get the price with max importance for each day
        idx = importance_df.groupby('date')['importance'].idxmax()
        daily_important = importance_df.loc[idx].set_index('date')['close_rounded']
        
        # Shift by 1 day to get PREVIOUS day's important price
        daily_important_shifted = daily_important.shift(1)
        
        # Map back to main dataframe
        self.df['prev_day_important_price'] = self.df['date'].map(daily_important_shifted)
        
        # Calculate distance index as percentage
        self.df['distance_index'] = ((self.df['close'] - self.df['prev_day_important_price']) / 
                                      (self.df['prev_day_important_price'].abs() + 1e-9)) * 100
        
        # 8. Delta T Acceleration (Derivative of DI) - New request
        # Vectorized diff (i - (i-1))
        self.df['delta_di'] = self.df['distance_index'].diff()
        
        
        # Cleanup temp column
        self.df.drop(columns=['close_rounded'], inplace=True, errors='ignore')
        
    def calculate_signals(self):
        """Calculates signals using vectorized operations (fast)"""
        if self.df is None or self.df.empty: return
        
        # Prepare signal columns
        self.df['buy_signal'] = False
        self.df['sell_signal'] = False
        
        if self.strategy_type == 'inside_bar':
            self._vectorized_inside_bar()
        elif self.strategy_type == 'micro_channel':
            self._vectorized_micro_channel()
        elif self.strategy_type == 'bull_bear':
            self._vectorized_bull_bear()
        elif self.strategy_type == 'bull_bear_bear':
            self._vectorized_bull_bear_bear()
        elif self.strategy_type == 'sequence_reversal':
            self._vectorized_sequence_reversal()
        elif self.strategy_type == 'sma_trend':
            self._vectorized_sma_trend()
        elif self.strategy_type == 'std_reversal':
            self._vectorized_std_reversal()
        elif self.strategy_type == 'sma_pullback':
            self._vectorized_sma_pullback()
        elif self.strategy_type == 'three_soldiers':
            self._vectorized_three_soldiers()
        elif self.strategy_type == 'breakout_momentum':
            self._vectorized_breakout_momentum()

    def _vectorized_inside_bar(self):
        # Shifted Series for vectorized comparison
        c = self.df['close']
        o = self.df['open']
        sma = self.df['sma20']
        
        c3, c2, c1 = c.shift(3), c.shift(2), c.shift(1)
        o2, o1 = o.shift(2), o.shift(1)
        sma3 = sma.shift(3)
        
        # Mother Bar (i-2) Body
        body_high_2 = np.maximum(o2, c2)
        body_low_2 = np.minimum(o2, c2)
        
        # Buy: Fractal (Up-Down), c3 > sma3, Inside Bar
        cond_fractal_buy = (c3 < c2) & (c2 > c1)
        cond_sma_buy = (c3 > sma3)
        cond_inside = (o1 >= body_low_2) & (o1 <= body_high_2) & (c1 >= body_low_2) & (c1 <= body_high_2)
        
        self.df.loc[cond_fractal_buy & cond_sma_buy & cond_inside, 'buy_signal'] = True
        
        # Sell: Fractal (Down-Up), c3 < sma3, Inside Bar
        cond_fractal_sell = (c3 > c2) & (c2 < c1)
        cond_sma_sell = (c3 < sma3)
        # Inside condition is same
        
        self.df.loc[cond_fractal_sell & cond_sma_sell & cond_inside, 'sell_signal'] = True

    def _vectorized_micro_channel(self):
        c = self.df['close']
        h = self.df['high']
        l = self.df['low']
        sma = self.df['sma20']
        
        # Shifts
        h4, h3, h2, h1 = h.shift(4), h.shift(3), h.shift(2), h.shift(1)
        l4, l1 = l.shift(4), l.shift(1)
        c1 = c.shift(1)
        sma1, sma2 = sma.shift(1), sma.shift(2)
        
        # Buy: Falling Highs (h4 > h3 > h2), Liquidity Grab (l1 < l4), Trend Up, Slope Up
        cond_falling = (h4 > h3) & (h3 > h2)
        cond_grab = (l1 < l4)
        cond_trend = (c1 > sma1)
        cond_slope = (sma1 > sma2)
        
        self.df.loc[cond_falling & cond_grab & cond_trend & cond_slope, 'buy_signal'] = True
        
        # Sell: Rising Highs (h3 < h2 < h1), Trend Down, Slope Down
        # Original code used: h[i-3] < h[i-2] < h[i-1]
        cond_rising = (h3 < h2) & (h2 < h1)
        cond_trend_sell = (c1 < sma1)
        cond_slope_sell = (sma1 < sma2)
        
        self.df.loc[cond_rising & cond_trend_sell & cond_slope_sell, 'sell_signal'] = True

    def _vectorized_bull_bear(self):
        c = self.df['close']
        o = self.df['open']
        l = self.df['low']
        h = self.df['high']
        sma = self.df['sma20']
        
        c2, c1 = c.shift(2), c.shift(1)
        o2, o1 = o.shift(2), o.shift(1)
        l3, l1 = l.shift(3), l.shift(1)
        h3, h1 = h.shift(3), h.shift(1)
        sma2 = sma.shift(2)
        
        # Buy: i-2 Bull, i-1 Bear, i-2 > SMA, l1 < l3
        cond_i2_bull = (c2 > o2)
        cond_i1_bear = (c1 < o1)
        cond_sma = (c2 > sma2)
        cond_liq = (l1 < l3)
        
        self.df.loc[cond_i2_bull & cond_i1_bear & cond_sma & cond_liq, 'buy_signal'] = True
        
        # Sell: i-2 Bear, i-1 Bull, i-2 < SMA, h1 > h3
        cond_i2_bear = (c2 < o2)
        cond_i1_bull = (c1 > o1)
        cond_sma_sell = (c2 < sma2) # note: original used sma[i-2] but let's check. Yes sma[i-2].
        cond_liq_sell = (h1 > h3)
        
        self.df.loc[cond_i2_bear & cond_i1_bull & (c2 < sma2) & cond_liq_sell, 'sell_signal'] = True

    def _vectorized_bull_bear_bear(self):
        c = self.df['close']
        o = self.df['open']
        sma = self.df['sma20']
        
        c3, c2, c1 = c.shift(3), c.shift(2), c.shift(1)
        o3, o2, o1 = o.shift(3), o.shift(2), o.shift(1)
        sma2 = sma.shift(2)
        
        # Buy: i-3 Bull, i-2 Bear, i-1 Bear, i-2 > SMA
        cond_buy = (c3 > o3) & (c2 < o2) & (c1 < o1) & (c2 > sma2)
        self.df.loc[cond_buy, 'buy_signal'] = True
        
        # Sell: i-3 Bear, i-2 Bull, i-1 Bull, i-2 < SMA
        cond_sell = (c3 < o3) & (c2 > o2) & (c1 > o1) & (c2 < sma2) 
        self.df.loc[cond_sell, 'sell_signal'] = True

    def _vectorized_sequence_reversal(self):
        c = self.df['close']
        o = self.df['open']
        sma = self.df['sma20']
        
        is_bull = (c > o).astype(int)
        is_bear = (c < o).astype(int)
        
        # Buy: 6 Bears (i-10 to i-5), 3 Bulls (i-4 to i-2), i-5 < SMA
        # Rolling sum over shifted series
        # bears check at i-5 (window 6): i-5..i-10
        bears_seq = is_bear.shift(5).rolling(6).sum() == 6
        # bulls check at i-2 (window 3): i-2..i-4
        bulls_seq = is_bull.shift(2).rolling(3).sum() == 3
        cond_sma = c.shift(5) < sma.shift(5)
        
        self.df.loc[bears_seq & bulls_seq & cond_sma, 'buy_signal'] = True
        
        # Sell: 6 Bulls (i-10 to i-5), 3 Bears (i-4 to i-2), i-5 > SMA
        bulls_seq_sell = is_bull.shift(5).rolling(6).sum() == 6
        bears_seq_sell = is_bear.shift(2).rolling(3).sum() == 3
        cond_sma_sell = c.shift(5) > sma.shift(5)
        
        self.df.loc[bulls_seq_sell & bears_seq_sell & cond_sma_sell, 'sell_signal'] = True

    def _vectorized_sma_trend(self):
        c = self.df['close']
        o = self.df['open']
        sma = self.df['sma20']
        
        c2, c1 = c.shift(2), c.shift(1)
        o1 = o.shift(1)
        sma2 = sma.shift(2)
        
        # Buy: i-2 > SMA, i-1 Bull
        cond_buy = (c2 > sma2) & (c1 > o1)
        self.df.loc[cond_buy, 'buy_signal'] = True
        
        # Sell: i-2 < SMA, i-1 Bear
        cond_sell = (c2 < sma2) & (c1 < o1)
        self.df.loc[cond_sell, 'sell_signal'] = True

    def _vectorized_std_reversal(self):
        if 'rev_std_buy_signal' not in self.df.columns: return
        
        # Buy: > 3 at i-1
        cond_buy = self.df['rev_std_buy_signal'].shift(1) > 3
        self.df.loc[cond_buy, 'buy_signal'] = True
        
        # Sell: > 3 at i-1
        cond_sell = self.df['rev_std_sell_signal'].shift(1) > 3
        self.df.loc[cond_sell, 'sell_signal'] = True

    def _vectorized_sma_pullback(self):
        c = self.df['close']
        l = self.df['low']
        h = self.df['high']
        sma = self.df['sma20']
        
        c5, c2, c1 = c.shift(5), c.shift(2), c.shift(1)
        l5 = l.shift(5)
        h5 = h.shift(5)
        sma5 = sma.shift(5)
        
        # Buy: l5 < sma5, c5 > sma5, c1 > c2
        cond_buy = (l5 < sma5) & (c5 > sma5) & (c1 > c2)
        self.df.loc[cond_buy, 'buy_signal'] = True
        
        # Sell: h5 > sma5, c5 < sma5, c1 < c2
        cond_sell = (h5 > sma5) & (c5 < sma5) & (c1 < c2)
        self.df.loc[cond_sell, 'sell_signal'] = True

    def _vectorized_three_soldiers(self):
        h = self.df['high']
        l = self.df['low']
        c = self.df['close']
        sma = self.df['sma20']
        
        h3, h2, h1 = h.shift(3), h.shift(2), h.shift(1)
        l3, l2, l1 = l.shift(3), l.shift(2), l.shift(1)
        c1 = c.shift(1)
        sma1 = sma.shift(1)
        
        # Buy: h3 < h2 < h1, c1 > sma1
        cond_buy = (h3 < h2) & (h2 < h1) & (c1 > sma1)
        self.df.loc[cond_buy, 'buy_signal'] = True
        
        # Sell: l3 > l2 > l1, c1 < sma1
        cond_sell = (l3 > l2) & (l2 > l1) & (c1 < sma1)
        self.df.loc[cond_sell, 'sell_signal'] = True

    def _vectorized_breakout_momentum(self):
        c = self.df['close']
        o = self.df['open']
        h = self.df['high']
        l = self.df['low']
        sma = self.df['sma20']
        
        c4, c3, c2, c1 = c.shift(4), c.shift(3), c.shift(2), c.shift(1)
        o4, o3, o2, o1 = o.shift(4), o.shift(3), o.shift(2), o.shift(1)
        h2, h1 = h.shift(2), h.shift(1)
        l2, l1 = l.shift(2), l.shift(1)
        sma1 = sma.shift(1)
        
        body3 = (c3 - o3).abs()
        body4 = (c4 - o4).abs()
        
        # Buy
        cond_1 = body3 > (3 * body4)
        cond_2 = c2 > o2
        cond_3 = h1 > h2
        cond_4 = c1 > o1
        cond_5 = c1 > sma1
        cond_6 = (c4 > o4) & (c3 < o3)
        self.df.loc[cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6, 'buy_signal'] = True
        
        # Sell
        cond_1_s = body3 > (3 * body4)
        cond_2_s = c2 < o2
        cond_3_s = l1 < l2
        cond_4_s = c1 < o1
        cond_5_s = c1 < sma1
        cond_6_s = (c4 < o4) & (c3 > o3)
        self.df.loc[cond_1_s & cond_2_s & cond_3_s & cond_4_s & cond_5_s & cond_6_s, 'sell_signal'] = True


        
    def check_buy_conditions(self, i):
        if self.strategy_type == 'inside_bar':
            return self.check_buy_conditions_inside_bar(i)
        elif self.strategy_type == 'micro_channel':
            return self.check_buy_conditions_micro(i)
        elif self.strategy_type == 'bull_bear':
            return self.check_buy_conditions_bull_bear(i)
        elif self.strategy_type == 'bull_bear_bear':
            return self.check_buy_conditions_bull_bear_bear(i)
        elif self.strategy_type == 'sequence_reversal':
            return self.check_buy_conditions_sequence_reversal(i)
        elif self.strategy_type == 'sma_trend':
            return self.check_buy_conditions_sma_trend(i)
        elif self.strategy_type == 'std_reversal':
            return self.check_buy_conditions_std_reversal(i)
        elif self.strategy_type == 'sma_pullback':
            return self.check_buy_conditions_sma_pullback(i)
        elif self.strategy_type == 'three_soldiers':
            return self.check_buy_conditions_three_soldiers(i)
        elif self.strategy_type == 'breakout_momentum':
            return self.check_buy_conditions_breakout_momentum(i)
        return False

    def check_sell_conditions(self, i):
        if self.strategy_type == 'inside_bar':
            return self.check_sell_conditions_inside_bar(i)
        elif self.strategy_type == 'micro_channel':
            return self.check_sell_conditions_micro(i)
        elif self.strategy_type == 'sequence_reversal':
            return self.check_sell_conditions_sequence_reversal(i)
        elif self.strategy_type == 'sma_trend':
            return self.check_sell_conditions_sma_trend(i)
        elif self.strategy_type == 'bull_bear_bear':
            return self.check_sell_conditions_bull_bear_bear(i)
        elif self.strategy_type == 'bull_bear':
            return self.check_sell_conditions_bull_bear(i)
        elif self.strategy_type == 'std_reversal':
            return self.check_sell_conditions_std_reversal(i)
        elif self.strategy_type == 'sma_pullback':
            return self.check_sell_conditions_sma_pullback(i)
        elif self.strategy_type == 'three_soldiers':
            return self.check_sell_conditions_three_soldiers(i)
        elif self.strategy_type == 'breakout_momentum':
            return self.check_sell_conditions_breakout_momentum(i)
        return False

    def check_buy_conditions_bull_bear_bear(self, i):
        # i-3 Alta, i-2 Baixa, i-1 Baixa
        if i < 3: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        cond_i3_bull = close[i-3] > open_price[i-3]
        cond_i2_bear = close[i-2] < open_price[i-2]
        cond_i1_bear = close[i-1] < open_price[i-1]
        
        return cond_i3_bull and cond_i2_bear and cond_i1_bear

    def check_sell_conditions_bull_bear_bear(self, i):
        # Opposite: i-3 Baixa, i-2 Alta, i-1 Alta
        if i < 3: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        cond_i3_bear = close[i-3] < open_price[i-3]
        cond_i2_bull = close[i-2] > open_price[i-2]
        cond_i1_bull = close[i-1] > open_price[i-1]
        
        # SMA Filter: Sell if Close < SMA
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        cond_sma = close[i-2] < sma20[i-2]
        
        return cond_i3_bear and cond_i2_bull and cond_i1_bull and cond_sma

    def check_buy_conditions_inside_bar(self, i):
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-3]): return False
        
        # V12 Logic: Simplified
        cond_fractal = close[i-3] < close[i-2] and close[i-2] > close[i-1]
        cond_sma = close[i-3] > sma20[i-3]
        
        body_high_i2 = max(open_price[i-2], close[i-2])
        body_low_i2 = min(open_price[i-2], close[i-2])
        cond_inside = (open_price[i-1] >= body_low_i2 and open_price[i-1] <= body_high_i2 and close[i-1] >= body_low_i2 and close[i-1] <= body_high_i2)
        
        return cond_fractal and cond_sma and cond_inside

    def check_sell_conditions_inside_bar(self, i):
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        sma20 = self.df['sma20'].values
        if pd.isna(sma20[i-3]): return False
        
        # V12 Logic: Simplified
        cond_fractal = close[i-3] > close[i-2] and close[i-2] < close[i-1]
        cond_sma = close[i-3] < sma20[i-3]
        
        body_high_i2 = max(open_price[i-2], close[i-2])
        body_low_i2 = min(open_price[i-2], close[i-2])
        cond_inside = (open_price[i-1] >= body_low_i2 and open_price[i-1] <= body_high_i2 and close[i-1] >= body_low_i2 and close[i-1] <= body_high_i2)
        
        return cond_fractal and cond_sma and cond_inside

    def check_buy_conditions_micro(self, i):
        if i < 20: return False
        close = self.df['close'].values
        low = self.df['low'].values
        sma20 = self.df['sma20'].values
        if pd.isna(sma20[i-3]): return False
        # Optimization: Falling Highs (Micro Canal de Baixa) + Liquidity Grab
        # User requested: High[i-4] < High[i-3] ... (3 Lows Caindo). "Caindo" implies Decreasing >.
        # Logic: High[i-4] > High[i-3] > High[i-2] (Falling Highs)
        # And: Low[i-1] < Low[i-4] (Liquidity Grab / Break of Structure Start)
        
        # Need i-4
        if i < 20: return False
        high = self.df['high'].values
        low = self.df['low'].values
        
        # "3 Highs Caindo" (Falling Highs)
        cond_falling_highs = high[i-4] > high[i-3] and high[i-3] > high[i-2]
        
        # Liquidity Grab
        cond_grab = low[i-1] < low[i-4]
        
        # Trend
        cond_trend = close[i-1] > sma20[i-1]
        cond_slope = sma20[i-1] > sma20[i-2]
        
        return cond_falling_highs and cond_grab and cond_trend and cond_slope

    def check_sell_conditions_micro(self, i):
        if i < 20: return False
        close = self.df['close'].values
        high = self.df['high'].values
        sma20 = self.df['sma20'].values
        if pd.isna(sma20[i-3]): return False
        cond_highs = high[i-3] < high[i-2] and high[i-2] < high[i-1]
        cond_trend = close[i-1] < sma20[i-1]
        cond_slope = sma20[i-1] < sma20[i-2]
        return cond_highs and cond_trend and cond_slope


    def check_buy_conditions_bull_bear(self, i):
        # i-2 Alta, i-1 Baixa
        if i < 2: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        
        cond_i2_bull = close[i-2] > open_price[i-2]
        cond_i1_bear = close[i-1] < open_price[i-1]
        cond_sma = close[i-2] > sma20[i-2]
        
        # Optimization Var 1: Liquidity Grab (Low[i-1] < Low[i-3])
        # A candle de baixa (sinal) deve ter violado a minima do candle anterior ao de alta (i-3)?
        # A logica testada foi Low[i-1] < Low[i-3].
        # Precisamos dos lows.
        low = self.df['low'].values
        # Ensure i-3 exists
        if i < 3: return False
        cond_liquidity = low[i-1] < low[i-3]
        
        base_condition = cond_i2_bull and cond_i1_bear and cond_sma and cond_liquidity
        
        return base_condition
        
    def check_sell_conditions_bull_bear(self, i):
        # Opposite: i-2 Baixa, i-1 Alta
        if i < 2: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        
        cond_i2_bear = close[i-2] < open_price[i-2]
        cond_i1_bull = close[i-1] > open_price[i-1]
        cond_sma = close[i-2] < sma20[i-2]
        
        # Optimization Var 1: Liquidity Grab (High[i-1] > High[i-3])
        high = self.df['high'].values
        if i < 3: return False
        cond_liquidity = high[i-1] > high[i-3]
        
        base_condition = cond_i2_bear and cond_i1_bull and cond_sma and cond_liquidity
        
        return base_condition

    def check_buy_conditions_bull_bear_bear(self, i):
        # i-3 Alta, i-2 Baixa, i-1 Baixa
        if i < 3: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        
        cond_i3_bull = close[i-3] > open_price[i-3]
        cond_i2_bear = close[i-2] < open_price[i-2]
        cond_i1_bear = close[i-1] < open_price[i-1]
        cond_sma = close[i-2] > sma20[i-2]
        
        return cond_i3_bull and cond_i2_bear and cond_i1_bear and cond_sma

    def check_sell_conditions_sequence_reversal(self, i):
        # 6 candles Alta (i-10 a i-5), 3 candles Baixa (i-4 a i-2) -> Venda em i (based on i-1 closed?)
        # Logic says: "i-3 de baixa i-2 de baixa i-3 de baixa" typo? assumed i-4, i-3, i-2 bears.
        # Checking i-10 to i-5 (Bull)
        if i < 11: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        
        # 6 Bulls: i-10 to i-5
        bulls = all(close[k] > open_price[k] for k in range(i-10, i-4))
        
        # 3 Bears: i-4 to i-2
        bears = all(close[k] < open_price[k] for k in range(i-4, i-1))
        
        # Trend Filter (Contra-Trend Sell needs to be below SMA? Or Above?)
        # User request: "ao contrário para venda" -> i-2 < SMA(20).
        # Update (User): i-5 > SMA(20) (Check if the top of the sequence was above SMA)
        cond_sma = close[i-5] > sma20[i-5]
        
        return bulls and bears and cond_sma

    def check_buy_conditions_sequence_reversal(self, i):
        # 6 candles Baixa (i-10 a i-5), 3 candles Alta (i-4 a i-2) -> Compra em i
        if i < 11: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        
        if pd.isna(self.df['sma20'].values[i-2]): return False
        sma20 = self.df['sma20'].values
        
        # 6 Bears: i-10 to i-5
        bears = all(close[k] < open_price[k] for k in range(i-10, i-4))
        
        # 3 Bulls: i-4 to i-2
        bulls = all(close[k] > open_price[k] for k in range(i-4, i-1))
        
        # Trend / Reversal Filter: Close > SMA
        # Symmetric Update: i-5 < SMA(20) (Check if the bottom of the sequence was below SMA)
        cond_sma = close[i-5] < sma20[i-5]
        
        return bears and bulls and cond_sma

    def check_buy_conditions_std_reversal(self, i):
        # Return True if > 3 consecutive (which means == 4 in a window of 4)
        if i < 5: return False
        val = self.df['rev_std_buy_signal'].values[i-1]
        if pd.isna(val): return False
        return val > 3

    def check_sell_conditions_std_reversal(self, i):
        # Return True if > 3 consecutive (which means == 4 in a window of 4)
        if i < 5: return False
        val = self.df['rev_std_sell_signal'].values[i-1]
        if pd.isna(val): return False
        return val > 3


        
    def check_sell_conditions_sma_trend(self, i):
        # i-2 abaixo da sma(20), i-1 de baixa
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-2]): return False
        
        # i-2 abaixo da SMA (close < sma or all < sma? users usually mean close)
        cond_i2_below_sma = close[i-2] < sma20[i-2]
        
        # i-1 de baixa
        cond_i1_bear = close[i-1] < open_price[i-1]
        
        return cond_i2_below_sma and cond_i1_bear

    def check_buy_conditions_sma_trend(self, i):
        # Opposite: i-2 acima da sma(20), i-1 de alta
        if i < 20: return False
        close = self.df['close'].values
        open_price = self.df['open'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-2]): return False
        
        cond_i2_above_sma = close[i-2] > sma20[i-2]
        cond_i1_bull = close[i-1] > open_price[i-1]
        
        return cond_i2_above_sma and cond_i1_bull

    def check_sell_conditions_sma_pullback(self, i):
        # Sell: Price > SMA20 at i-5 AND Close i-1 < SMA20 
        # (Simplified pullback logic: Was above, now below)
        if i < 5: return False
        
        close = self.df['close'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i]): return False
        
        was_above = close[i-5] > sma20[i-5]
        is_below = close[i-1] < sma20[i-1]
        
        return was_above and is_below

    def check_buy_conditions_three_soldiers(self, i):
        # 3 Soldados de Alta
        # 1. Highs ascendentes: high[i-3] < high[i-2] < high[i-1]
        # 2. Tendência Bull: close[i-1] > SMA(20)
        if i < 4: return False
        
        high = self.df['high'].values
        close = self.df['close'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-1]): return False
        
        # Highs ascendentes
        cond_highs = (high[i-3] < high[i-2]) and (high[i-2] < high[i-1])
        
        # Tendência
        cond_trend = close[i-1] > sma20[i-1]
        
        # Opcional: i-1 tem que ser candle de alta? Pela descrição não foi explícito, 
        # mas "SMA > 20" sugere compra
        
        return cond_highs and cond_trend

    def check_sell_conditions_three_soldiers(self, i):
        # 3 Soldados de Baixa (Three Black Crows logic adaptada)
        # 1. Lows descendentes: low[i-3] > low[i-2] > low[i-1]
        # 2. Tendência Bear: close[i-1] < SMA(20)
        if i < 4: return False
        
        low = self.df['low'].values
        close = self.df['close'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-1]): return False
        
        # Lows descendentes
        cond_lows = (low[i-3] > low[i-2]) and (low[i-2] > low[i-1])
        
        # Tendência
        cond_trend = close[i-1] < sma20[i-1]
        
        return cond_lows and cond_trend

    def check_buy_conditions_breakout_momentum(self, i):
        # Estratégia Breakout Momentum (Compra)
        # 1. Corpo i-3 > 3x Corpo i-4
        # 2. i-2 é de Alta (close > open)
        # 3. High i-1 > High i-2
        # 4. i-1 é de Alta
        # 5. Close i-1 > SMA(20)
        if i < 5: return False
        
        close = self.df['close'].values
        open_p = self.df['open'].values
        high = self.df['high'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-1]): return False
        
        # Tamanho dos corpos
        body_i3 = abs(close[i-3] - open_p[i-3])
        body_i4 = abs(close[i-4] - open_p[i-4])
        
        # 6. Direção dos candles (Invertido a pedido do usuario):
        #    i-4 deve ser de ALTA (preparação)
        #    i-3 deve ser de BAIXA (explosão - Dip Buying?)
        
        cond_1 = body_i3 > (3 * body_i4)
        cond_2 = close[i-2] > open_p[i-2]
        cond_3 = high[i-1] > high[i-2]
        cond_4 = close[i-1] > open_p[i-1]
        cond_5 = close[i-1] > sma20[i-1]
        
        # Direção i-4 (Alta) e i-3 (Baixa)
        cond_6 = (close[i-4] > open_p[i-4]) and (close[i-3] < open_p[i-3])
        
        return cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and cond_6

    def check_sell_conditions_breakout_momentum(self, i):
        # Estratégia Breakout Momentum (Venda)
        # 1. Corpo i-3 > 3x Corpo i-4
        # 2. i-2 é de Baixa (close < open)
        # 3. Low i-1 < Low i-2
        # 4. i-1 é de Baixa
        # 5. Close i-1 < SMA(20)
        if i < 5: return False
        
        close = self.df['close'].values
        open_p = self.df['open'].values
        low = self.df['low'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-1]): return False
        
        # Tamanho dos corpos
        body_i3 = abs(close[i-3] - open_p[i-3])
        body_i4 = abs(close[i-4] - open_p[i-4])
        
        # 6. Direção dos candles (Invertido):
        #    i-4 deve ser de BAIXA (preparação)
        #    i-3 deve ser de ALTA (explosão - Top Selling?)
        
        cond_1 = body_i3 > (3 * body_i4)
        cond_2 = close[i-2] < open_p[i-2]
        cond_3 = low[i-1] < low[i-2]
        cond_4 = close[i-1] < open_p[i-1]
        cond_5 = close[i-1] < sma20[i-1]
        
        # Direção i-4 (Baixa) e i-3 (Alta)
        cond_6 = (close[i-4] < open_p[i-4]) and (close[i-3] > open_p[i-3])
        
        return cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and cond_6

    def check_buy_conditions_sma_pullback(self, i):
        # low[i-5] < sma(20); close[i-5] > sma(20); close[i-1] > close[i-2]
        if i < 20: return False
        close = self.df['close'].values
        low = self.df['low'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-5]): return False
        
        cond_pullback = low[i-5] < sma20[i-5] and close[i-5] > sma20[i-5]
        cond_trigger = close[i-1] > close[i-2]
        
        return cond_pullback and cond_trigger

    def check_sell_conditions_sma_pullback(self, i):
        # high[i-5] > sma(20); close[i-5] < sma(20); close[i-1] < close[i-2]
        if i < 20: return False
        close = self.df['close'].values
        high = self.df['high'].values
        sma20 = self.df['sma20'].values
        
        if pd.isna(sma20[i-5]): return False
        
        cond_pullback = high[i-5] > sma20[i-5] and close[i-5] < sma20[i-5]
        cond_trigger = close[i-1] < close[i-2]
        
        return cond_pullback and cond_trigger

    def calculate_angle(self, i):
        if i < 5: return 0
        sma = self.df['sma20'].values
        atr = self.df['atr'].values
        
        dy = sma[i] - sma[i-5]
        avg_atr = atr[i] if not pd.isna(atr[i]) and atr[i] > 0 else 1.0
        
        # Normalize slope by ATR:
        # Slope (points per bar) = dy / 5
        # Normalized Slope (ATRs per bar) = Slope / avg_atr
        # Scaling Factor 10 makes 0.1 ATR/bar -> 45 degrees (Visual Scaling)
        
        normalized_slope = ((dy / 5) / avg_atr) * 10
        return math.degrees(math.atan(normalized_slope))

    def simulate_buy_trade(self, entry_idx, tp_mult, sl_mult, ratio_label):
        entry_price = self.df['close'].values[entry_idx - 1] # Assumindo entrada no fechamento de i-1 (sinal confirmado)
        entry_time = self.df['time'].values[entry_idx - 1]
        
        if self.strategy_type == 'micro_channel':
            candle_range = self.df['high'].values[entry_idx - 1] - self.df['low'].values[entry_idx - 1]
            if candle_range <= 0: candle_range = 5
            target_price = entry_price + (candle_range * tp_mult)
            stop_price = entry_price - (candle_range * sl_mult) 
            # Nota: user pediu Stop SL x Range. Se fosse técnico: stop = low[i-1].
            # Mantendo logica de multiplier
        else:
            candle_range = self.df['high'].values[entry_idx - 2] - self.df['low'].values[entry_idx - 2]
            target_price = entry_price + (candle_range * tp_mult)
            stop_price = entry_price - (candle_range * sl_mult)

        volatility = 0 # Simplificado para performance do loop principal, calculado no run_backtest depois se precisar
        # Mas simulate precisa retornar volatilidade.
        # Vamos calcular range medio rapido?
        # volatility = self.df['range'].values[entry_idx-21:entry_idx-1].mean()
        # Isso pode ser lento em loop. Vamos deixar volatility como N/A e calcular no batch se der.
        # Ou calcular agora mesmo. Slice numpy é rapido.
        volatility = np.mean(self.df['range'].values[max(0, entry_idx-21):entry_idx-1])

        close_i2 = self.df['close'].values[entry_idx - 2]
        sma_i2 = self.df['sma20'].values[entry_idx - 2]
        dist_sma_i2 = abs(close_i2 - sma_i2) if not pd.isna(sma_i2) else 0.0

        # Loop for exit
        headers = self.df['high'].values
        lowers = self.df['low'].values
        closes = self.df['close'].values
        # Slice for performance
        search_max = min(len(self.df), entry_idx + 500) # Max hold time assumption? Nao, until stop/loss.
        # Mas para 2M candles, iterar até o fim é O(N^2).
        # Vamos assumir que sai em até 1000 candles.
        
        exit_price = closes[-1]
        result = 0
        winner = False
        exit_bars = 0
        
        # Numba Optimized Search
        exit_price, winner, exit_bars = find_exit_buy_numba(headers, lowers, closes, entry_idx, target_price, stop_price)
            
        result = exit_price - entry_price
        
        # Slope details
        slope_deg = self.calculate_angle(entry_idx)

        return {
            'entry_time': entry_time,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'result': float(result),
            'winner': winner,
            'direction': 'Compra',
            'hour': pd.Timestamp(entry_time).hour,
            'weekday': pd.Timestamp(entry_time).dayofweek,
            'year': pd.Timestamp(entry_time).year,
            'month': pd.Timestamp(entry_time).month,
            'year_month': pd.Timestamp(entry_time).strftime('%Y-%m'),
            'volatility': float(volatility),
            'exit_bars': exit_bars,
            'entry_idx': int(entry_idx),
            'target_price': float(target_price),
            'stop_price': float(stop_price),
            'tp_points': float(target_price - entry_price),
            'sl_points': float(entry_price - stop_price),
            'rr_ratio': float((target_price - entry_price) / (entry_price - stop_price)) if (entry_price - stop_price) != 0 else 0,
            'ratio_label': ratio_label,
            'slope_degrees': float(slope_deg),
            'dist_sma_i2': float(dist_sma_i2),
            'distance_index_i2': float(self.df['distance_index'].values[entry_idx - 2]) if not pd.isna(self.df['distance_index'].values[entry_idx - 2]) else 0.0,
            'delta_di_i2': float(self.df['delta_di'].values[entry_idx - 2]) if not pd.isna(self.df['delta_di'].values[entry_idx - 2]) else 0.0
        }

    def simulate_sell_trade(self, entry_idx, tp_mult, sl_mult, ratio_label):
        entry_price = self.df['close'].values[entry_idx - 1]
        entry_time = self.df['time'].values[entry_idx - 1]
        
        if self.strategy_type == 'micro_channel':
            candle_range = self.df['high'].values[entry_idx - 1] - self.df['low'].values[entry_idx - 1]
            if candle_range <= 0: candle_range = 5
            target_price = entry_price - (candle_range * tp_mult)
            stop_price = entry_price + (candle_range * sl_mult)
        else:
            candle_range = self.df['high'].values[entry_idx - 2] - self.df['low'].values[entry_idx - 2]
            target_price = entry_price - (candle_range * tp_mult)
            stop_price = entry_price + (candle_range * sl_mult)

        volatility = np.mean(self.df['range'].values[max(0, entry_idx-21):entry_idx-1])

        close_i2 = self.df['close'].values[entry_idx - 2]
        sma_i2 = self.df['sma20'].values[entry_idx - 2]
        dist_sma_i2 = abs(close_i2 - sma_i2) if not pd.isna(sma_i2) else 0.0

        headers = self.df['high'].values
        lowers = self.df['low'].values
        closes = self.df['close'].values
        
        exit_price = closes[-1]
        result = 0
        winner = False
        exit_bars = 0
        
        # Numba Optimized Search
        exit_price, winner, exit_bars = find_exit_sell_numba(headers, lowers, closes, entry_idx, target_price, stop_price)
            
        result = entry_price - exit_price
        slope_deg = self.calculate_angle(entry_idx)

        return {
            'entry_time': entry_time,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'result': float(result),
            'winner': winner,
            'direction': 'Venda',
            'hour': pd.Timestamp(entry_time).hour,
            'weekday': pd.Timestamp(entry_time).dayofweek,
            'year': pd.Timestamp(entry_time).year,
            'month': pd.Timestamp(entry_time).month,
            'year_month': pd.Timestamp(entry_time).strftime('%Y-%m'),
            'volatility': float(volatility),
            'exit_bars': exit_bars,
            'entry_idx': int(entry_idx),
            'target_price': float(target_price),
            'stop_price': float(stop_price),
            'tp_points': float(entry_price - target_price),
            'sl_points': float(stop_price - entry_price),
            'rr_ratio': float((entry_price - target_price) / (stop_price - entry_price)) if (stop_price - entry_price) != 0 else 0,
            'ratio_label': ratio_label,
            'slope_degrees': float(slope_deg),
            'dist_sma_i2': float(dist_sma_i2),
            'distance_index_i2': float(self.df['distance_index'].values[entry_idx - 2]) if not pd.isna(self.df['distance_index'].values[entry_idx - 2]) else 0.0,
            'delta_di_i2': float(self.df['delta_di'].values[entry_idx - 2]) if not pd.isna(self.df['delta_di'].values[entry_idx - 2]) else 0.0
        }

    def run_backtest(self):
        # Tenta carregar do cache
        if self.load_cache():
            self.set_active_ratio(self.AVAILABLE_RATIOS[2]['label'])
            print("Loaded from Cache.")
            return

        self.load_data() # Load heavy data only if needed

        # 1. Calculate Signals using Vectorized Logic
        print("Finding signals (Vectorized)...")
        start_time = datetime.now()
        self.calculate_signals()
        
        # 2. Pre-calculate metrics for all rows to allow fast indexing
        # Volatility (Mean range of last 20 candles ending at i-1)
        # Shift(1) gives range at i-1. Rolling(20).mean() gives mean of i-20..i-1
        self.df['volatility_calc'] = self.df['range'].shift(1).rolling(window=20).mean()
        
        # Slope Degrees
        # calculate_angle logic: dy = sma - sma(i-5). 
        dy = self.df['sma20'] - self.df['sma20'].shift(5)
        avg_atr = self.df['atr'].fillna(1.0)
        norm_slope = ((dy / 5) / avg_atr) * 10
        self.df['slope_deg'] = np.degrees(np.arctan(norm_slope))
        
        # Dist SMA (Points, Absolute, at i-2)
        self.df['dist_sma_points'] = (self.df['close'] - self.df['sma20']).abs()
        
        # Shifts for i-2 metrics (used in results)
        self.df['dist_sma_points_i2'] = self.df['dist_sma_points'].shift(2)
        self.df['distance_index_i2'] = self.df['distance_index'].shift(2)
        self.df['delta_di_i2'] = self.df['delta_di'].shift(2)
        
        # Volume SMA and Slope
        if 'tick_volume' in self.df.columns:
            self.df['vol_sma20'] = self.df['tick_volume'].rolling(window=20).mean()
            self.df['vol_slope'] = self.df['vol_sma20'].diff()
            # Capture for i-2 (signal time reference)
            self.df['vol_slope_i2'] = self.df['vol_slope'].shift(2)
        else:
            self.df['vol_slope_i2'] = 0.0
            
        # Jerk (Difference of Delta T) = Diff of Delta DI
        self.df['jerk_di'] = self.df['delta_di'].diff()
        self.df['jerk_di_i2'] = self.df['jerk_di'].shift(2)
        
        # Filter for signals
        # We need to drop NaNs created by shifts to avoid issues, or handled by subset
        signal_mask = (self.df['buy_signal'] | self.df['sell_signal'])
        # Limit start index to 20 as per original
        signal_mask.iloc[:20] = False
        
        signals_df = self.df.loc[signal_mask].copy()
        
        print(f"Found {len(signals_df)} signals. Calculating all ratios (Vectorized)...")
        
        if signals_df.empty:
            self.all_trades_df = pd.DataFrame()
            self.save_cache()
            return
            
        # Prepare arrays for Numba
        entry_indices = signals_df.index.values
        # Directions: 1 for Buy, -1 for Sell
        directions = np.where(signals_df['buy_signal'], 1, -1).astype(np.int32)
        
        # Ranges for TP/SL Calculation
        # Micro Channel: Range at i-1
        # Others: Range at i-2
        if self.strategy_type == 'micro_channel':
            ranges = self.df['range'].shift(1).loc[signal_mask].values
        else:
            ranges = self.df['range'].shift(2).loc[signal_mask].values
            
        # Ensure range > 0 (fallback to 5 as in original)
        ranges = np.where(ranges <= 0, 5.0, ranges)
        
        # Data Arrays for Simulation
        opens_arr = self.df['open'].values
        highs_arr = self.df['high'].values
        lows_arr = self.df['low'].values
        closes_arr = self.df['close'].values
        
        all_dfs = []
        
        for ratio in self.AVAILABLE_RATIOS:
            tp_mult = float(ratio['tp'])
            sl_mult = float(ratio['sl'])
            ratio_label = ratio['label']
            
            # Batch Simulation
            results, winners, exit_prices, exit_bars, targets, stops = simulate_trades_batch_numba(
                entry_indices, directions, opens_arr, highs_arr, lows_arr, closes_arr, ranges, tp_mult, sl_mult
            )
            
            # Create DataFrame for this ratio
            ratio_df = signals_df.copy()
            ratio_df['entry_idx'] = entry_indices
            ratio_df['entry_price'] = closes_arr[entry_indices - 1] # Entry at Close[i-1] matches original
            ratio_df['entry_time'] = self.df.loc[entry_indices - 1, 'time'].values # Entry time at i-1
            # Note: signals_df index is 'i'. original 'entry_time' was df['time'].values[entry_idx-1].
            # signals_df['time'] is time at 'i'. We want time at 'i-1'.
            
            ratio_df['direction'] = np.where(directions == 1, 'Compra', 'Venda')
            ratio_df['result'] = results
            ratio_df['winner'] = winners
            ratio_df['exit_price'] = exit_prices
            ratio_df['exit_bars'] = exit_bars
            ratio_df['target_price'] = targets
            ratio_df['stop_price'] = stops
            ratio_df['ratio_label'] = ratio_label
            ratio_df['volatility'] = ratio_df['volatility_calc']
            ratio_df['dist_sma_i2'] = ratio_df['dist_sma_points_i2']
            
            # Slope degrees already in signals_df (calculated for all)
            ratio_df['slope_degrees'] = ratio_df['slope_deg']
            
            # Calculate Points
            # For Buy: result = exit - entry. TP points = target - entry
            # For Sell: result = entry - exit. TP points = entry - target
            ratio_df['tp_points'] = np.where(directions == 1, ratio_df['target_price'] - ratio_df['entry_price'], ratio_df['entry_price'] - ratio_df['target_price'])
            ratio_df['sl_points'] = np.where(directions == 1, ratio_df['entry_price'] - ratio_df['stop_price'], ratio_df['stop_price'] - ratio_df['entry_price'])
            
            # RR Ratio
            # Avoid div by zero
            sl_p = ratio_df['sl_points']
            ratio_df['rr_ratio'] = np.where(sl_p != 0, ratio_df['tp_points'] / sl_p, 0.0)
            
            # Keep only necessary columns
            keep_cols = ['entry_time', 'entry_price', 'exit_price', 'result', 'winner', 'direction', 
                         'hour', 'weekday', 'year', 'month', 'year_month', 'volatility', 'exit_bars', 
                         'entry_idx', 'target_price', 'stop_price', 'tp_points', 'sl_points', 'rr_ratio',
                         'ratio_label', 'slope_degrees', 'dist_sma_i2', 'distance_index_i2', 'delta_di_i2',
                         'vol_slope', 'jerk_di']
            
            # Map i2 cols
            ratio_df['vol_slope'] = ratio_df['vol_slope_i2']
            ratio_df['jerk_di'] = ratio_df['jerk_di_i2']
            
            all_dfs.append(ratio_df[keep_cols])

        if all_dfs:
            self.all_trades_df = pd.concat(all_dfs, ignore_index=True)
            
            # Levels Calculation (Vectorized on full result)
            try:
                self.all_trades_df['vol_level'] = pd.qcut(self.all_trades_df['volatility'], q=5, labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'], duplicates='drop').astype(str)
            except: self.all_trades_df['vol_level'] = 'Média'
            
            try:
                self.all_trades_df['dist_level'] = pd.qcut(self.all_trades_df['dist_sma_i2'], q=10, labels=[f'D{i}' for i in range(1, 11)], duplicates='drop').astype(str)
            except: self.all_trades_df['dist_level'] = 'D5'
            
            try:
                self.all_trades_df['di_level'] = pd.qcut(self.all_trades_df['distance_index_i2'].abs(), q=10, labels=[f'DI{i}' for i in range(1, 11)], duplicates='drop').astype(str)
            except: self.all_trades_df['di_level'] = 'DI5'
            
            try:
                self.all_trades_df['acc_level'] = pd.qcut(self.all_trades_df['delta_di_i2'], q=20, labels=[f'ACC{i}' for i in range(1, 21)], duplicates='drop').astype(str)
            except: self.all_trades_df['acc_level'] = 'ACC10'
            
            # Volume Slope Levels (5 bins)
            try:
                self.all_trades_df['vol_slope_level'] = pd.qcut(self.all_trades_df['vol_slope'], q=5, labels=['Queda Forte', 'Queda Leve', 'Flat', 'Subida Leve', 'Subida Forte'], duplicates='drop').astype(str)
            except: self.all_trades_df['vol_slope_level'] = 'Flat'
            
            # Jerk Levels (Difference of Delta T - 20 bins)
            try:
                self.all_trades_df['jerk_level'] = pd.qcut(self.all_trades_df['jerk_di'], q=20, labels=[f'J{i}' for i in range(1, 21)], duplicates='drop').astype(str)
            except: self.all_trades_df['jerk_level'] = 'J10'
        else:
             self.all_trades_df = pd.DataFrame()
             
        end_time = datetime.now()
        print(f"Backtest finished in {(end_time - start_time).total_seconds():.2f}s")
        self.save_cache()
        self.set_active_ratio(self.AVAILABLE_RATIOS[2]['label'])

    def set_active_ratio(self, ratio_label):
        self.current_ratio = ratio_label
        if not self.all_trades_df.empty:
            self.trades_df = self.all_trades_df[self.all_trades_df['ratio_label'] == ratio_label].copy()
            if not self.trades_df.empty:
                self.trades_df['cumulative_result'] = self.trades_df['result'].cumsum()
            self.trades = self.trades_df.to_dict('records')
        else:
            self.trades_df = pd.DataFrame()
            self.trades = []
            
        self.reset_filters()
        return len(self.trades)

    def apply_filters(self, filters):
        if self.trades_df.empty: return
        self.filtered_df = self.trades_df.copy()
        
        if filters.get('direction') and filters['direction'] != 'all':
            self.filtered_df = self.filtered_df[self.filtered_df['direction'] == filters['direction']]
        if filters.get('hours'):
            self.filtered_df = self.filtered_df[self.filtered_df['hour'].isin(filters['hours'])]
        if filters.get('weekdays'):
            self.filtered_df = self.filtered_df[self.filtered_df['weekday'].isin(filters['weekdays'])]
        if filters.get('years'):
            self.filtered_df = self.filtered_df[self.filtered_df['year'].isin(filters['years'])]
        if filters.get('volatility'):
            if 'vol_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['vol_level'].isin(filters['volatility'])]
        if filters.get('dist_levels'):
            if 'dist_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['dist_level'].isin(filters['dist_levels'])]
        if filters.get('angle_ranges'):
            bins = list(range(0, 95, 5)) + [1000]
            labels = [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+']
            self.filtered_df['angle_bin'] = pd.cut(self.filtered_df['slope_degrees'].abs(), bins=bins, labels=labels, right=False)
            self.filtered_df = self.filtered_df[self.filtered_df['angle_bin'].isin(filters['angle_ranges'])]
        if filters.get('di_levels'):
            if 'di_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['di_level'].isin(filters['di_levels'])]
        if filters.get('acc_levels'):
            if 'acc_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['acc_level'].isin(filters['acc_levels'])]
        if filters.get('acc_levels'):
            if 'acc_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['acc_level'].isin(filters['acc_levels'])]
        
        # New Filters
        if filters.get('vol_slope_levels'):
            if 'vol_slope_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['vol_slope_level'].isin(filters['vol_slope_levels'])]
        
        if filters.get('jerk_levels'):
            if 'jerk_level' in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[self.filtered_df['jerk_level'].isin(filters['jerk_levels'])]
            
        if filters.get('invert'):
            self.filtered_df['result'] = -self.filtered_df['result']
            self.filtered_df['winner'] = ~self.filtered_df['winner']
            
        if not self.filtered_df.empty:
            self.filtered_df['cumulative_result'] = self.filtered_df['result'].cumsum()

    def get_result_for_filters(self, ratio_label, filters):
        if self.all_trades_df.empty: return pd.DataFrame()
        df = self.all_trades_df[self.all_trades_df['ratio_label'] == ratio_label].copy()
        # Re-apply same logic as apply_filters but on local df... (omitted for brevity, assume similar logic or refactor)
        # Simplified for now:
        if filters.get('direction') and filters['direction'] != 'all': df = df[df['direction'] == filters['direction']]
        if filters.get('hours'): df = df[df['hour'].isin(filters['hours'])]
        if filters.get('weekdays'): df = df[df['weekday'].isin(filters['weekdays'])]
        if filters.get('years'): df = df[df['year'].isin(filters['years'])]
        if filters.get('volatility') and 'vol_level' in df.columns: df = df[df['vol_level'].isin(filters['volatility'])]
        if filters.get('dist_levels') and 'dist_level' in df.columns: df = df[df['dist_level'].isin(filters['dist_levels'])]
        if filters.get('angle_ranges'):
             bins = list(range(0, 95, 5)) + [1000]
             labels = [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+']
             df['angle_bin'] = pd.cut(df['slope_degrees'].abs(), bins=bins, labels=labels, right=False)
             df = df[df['angle_bin'].isin(filters['angle_ranges'])]
        if filters.get('di_levels') and 'di_level' in df.columns: df = df[df['di_level'].isin(filters['di_levels'])]
        if filters.get('acc_levels') and 'acc_level' in df.columns: df = df[df['acc_level'].isin(filters['acc_levels'])]
        if filters.get('vol_slope_levels') and 'vol_slope_level' in df.columns: df = df[df['vol_slope_level'].isin(filters['vol_slope_levels'])]
        if filters.get('jerk_levels') and 'jerk_level' in df.columns: df = df[df['jerk_level'].isin(filters['jerk_levels'])]
        if filters.get('invert'):
            df['result'] = -df['result']
            df['winner'] = ~df['winner']
        return df

    def set_combined_filtered_df(self, strategies_data):
        dfs = []
        for s in strategies_data:
            dfs.append(self.get_result_for_filters(s.get('ratio_label'), s.get('filters', {})))
        if dfs:
            final = pd.concat(dfs).drop_duplicates(subset=['entry_idx', 'direction', 'ratio_label']).sort_values('entry_time')
            final['cumulative_result'] = final['result'].cumsum()
            self.filtered_df = final
            return len(final)
        self.filtered_df = pd.DataFrame()
        return 0

    def reset_filters(self):
        self.filtered_df = self.trades_df.copy()

    def calculate_sharpe(self, returns, risk_free=0.0):
        if len(returns) < 2: return 0
        if returns.std() == 0: return 0
        return ((returns.mean() - risk_free) / returns.std()) * np.sqrt(252)

    def get_summary(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return {'total_trades': 0, 'error': 'No trades'}
        
        # Recalculate basic stats on demand
        total = len(df)
        winners = df['winner'].sum()
        losers = total - winners
        win_rate = (winners/total)*100
        total_profit = df['result'].sum()
        max_dd = 0
        
        # Fast calc drawdown
        creturn = df['result'].cumsum().values
        if len(creturn) > 0:
            running_max = np.maximum.accumulate(creturn)
            drawdowns = running_max - creturn
            max_dd = drawdowns.max()

        # Sharpe
        sharpe = self.calculate_sharpe(df['result'])
        
        # Profit Factor
        gross_profit = df[df['winner']]['result'].sum()
        gross_loss = abs(df[~df['winner']]['result'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0

        # Dates
        data_start = df['entry_time'].min()
        data_end = df['entry_time'].max()
        
        return {
            'total_trades': int(total),
            'winners': int(winners),
            'losers': int(losers),
            'win_rate': float(round(win_rate, 2)),
            'total_profit_points': float(round(total_profit, 2)),
            'avg_profit_per_trade': float(round(df['result'].mean(), 2)),
            'max_drawdown': float(round(max_dd, 2)),
            'sharpe_ratio': float(round(sharpe, 2)),
            'buy_trades': int(len(df[df['direction']=='Compra'])),
            'sell_trades': int(len(df[df['direction']=='Venda'])),
            'avg_winner': float(round(df[df['winner']]['result'].mean(), 2)) if winners > 0 else 0,
            'avg_loser': float(round(df[~df['winner']]['result'].mean(), 2)) if losers > 0 else 0,
            'profit_factor': float(round(profit_factor, 2)),
            'data_start': data_start.strftime('%d/%m/%Y') if not pd.isna(data_start) else '-',
            'data_end': data_end.strftime('%d/%m/%Y') if not pd.isna(data_end) else '-'
        }

    def get_stats_by_hour(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        stats = []
        for h in sorted(df['hour'].unique()):
            d = df[df['hour']==h]
            stats.append({
                'hour': int(h), 
                'trades': len(d), 
                'win_rate': float(round((d['winner'].sum()/len(d))*100, 2)), 
                'total_profit': float(round(d['result'].sum(), 2)),
                'avg_profit': float(round(d['result'].mean(), 2)),
                'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
            })
        return stats

    def get_stats_by_weekday(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        stats = []
        # Iterate Mon-Fri (0-4) explicitly to ensure chart consistency
        for w in range(5):
            d = df[df['weekday']==w]
            if d.empty:
                stats.append({
                    'weekday_name': names[w], 
                    'weekday': int(w),
                    'trades': 0, 
                    'win_rate': 0.0, 
                    'total_profit': 0.0,
                    'avg_profit': 0.0,
                    'sharpe_ratio': 0.0
                })
            else:
                stats.append({
                    'weekday_name': names[w], 
                    'weekday': int(w),
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d))*100, 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats
        
    def get_stats_by_year(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        stats = []
        for y in sorted(df['year'].unique()):
            d = df[df['year']==y]
            stats.append({
                'year': int(y), 
                'trades': len(d), 
                'win_rate': float(round((d['winner'].sum()/len(d))*100, 2)), 
                'total_profit': float(round(d['result'].sum(), 2)),
                'avg_profit': float(round(d['result'].mean(), 2)),
                'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
            })
        return stats

    def get_stats_by_month(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        stats = []
        for m in sorted(df['year_month'].unique()):
            d = df[df['year_month']==m]
            stats.append({
                'year_month': str(m), 
                'trades': len(d), 
                'win_rate': float(round((d['winner'].sum()/len(d))*100, 2)), 
                'total_profit': float(round(d['result'].sum(), 2)),
                'avg_profit': float(round(d['result'].mean(), 2)),
                'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
            })
        return stats

    def get_stats_by_volatility(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty or 'vol_level' not in df.columns: return []
        stats = []
        for v in ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']:
            d = df[df['vol_level'].astype(str)==v]
            if len(d) > 0:
                stats.append({
                    'volatility': v, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_dist_sma(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty or 'dist_level' not in df.columns: return []
        stats = []
        for l in [f'D{i}' for i in range(1, 11)]:
            d = df[df['dist_level'].astype(str)==l]
            if len(d) > 0:
                stats.append({
                    'dist_level': l, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_di(self, use_filtered=False):
        """Statistics by Distance Index (DI1-DI10) - distance from previous day's important price"""
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty or 'di_level' not in df.columns: return []
        stats = []
        for l in [f'DI{i}' for i in range(1, 11)]:
            d = df[df['di_level'].astype(str)==l]
            if len(d) > 0:
                stats.append({
                    'di_level': l, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats
        
    def get_stats_by_acc(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        # Predefined order ACC1 to ACC20
        labels = [f'ACC{i}' for i in range(1, 21)]
        stats = []
        for l in labels:
            d = df[df['acc_level']==l]
            if len(d) > 0:
                stats.append({
                    'acc_level': l,
                    'trades': len(d),
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)),
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_vol_slope(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty or 'vol_slope_level' not in df.columns: return []
        # Custom sort order
        order = ['Queda Forte', 'Queda Leve', 'Flat', 'Subida Leve', 'Subida Forte']
        stats = []
        for l in order:
            d = df[df['vol_slope_level'].astype(str)==l]
            if len(d) > 0:
                stats.append({
                    'vol_slope_level': l,
                    'trades': len(d),
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)),
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_jerk(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        # Predefined order J1 to J20
        labels = [f'J{i}' for i in range(1, 21)]
        stats = []
        for l in labels:
            d = df[df['jerk_level'].astype(str)==l]
            if len(d) > 0:
                stats.append({
                    'jerk_level': l,
                    'trades': len(d),
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)),
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats
        
    def get_stats_by_acc(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        # Predefined order ACC1 to ACC20
        labels = [f'ACC{i}' for i in range(1, 21)]
        stats = []
        for l in labels:
            d = df[df['acc_level']==l]
            if len(d) > 0:
                stats.append({
                    'acc_level': l,
                    'trades': len(d),
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)),
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_angle(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        bins = list(range(0, 95, 5)) + [1000]
        labels = [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+']
        df['angle_bin'] = pd.cut(df['slope_degrees'].abs(), bins=bins, labels=labels, right=False)
        stats = []
        for l in labels:
            d = df[df['angle_bin']==l]
            if len(d) > 0:
                stats.append({
                    'angle_range': l, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats
    
    def get_stats_by_take_stop(self, use_filtered=False):
        if self.all_trades_df.empty: return []
        
        # Filter all_trades_df correctly if use_filtered is True
        target_df = self.all_trades_df
        if use_filtered and not self.filtered_df.empty:
            valid_indices = self.filtered_df['entry_idx'].unique()
            target_df = self.all_trades_df[self.all_trades_df['entry_idx'].isin(valid_indices)]

        stats = []
        for ratio in self.AVAILABLE_RATIOS:
            label = ratio['label']
            d = target_df[target_df['ratio_label']==label]
            if len(d)>0: 
                stats.append({
                    'take_stop_type': label, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'avg_rr': float(round(d['rr_ratio'].mean(), 2)) if 'rr_ratio' in d.columns else 0.0,
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
        return stats

    def get_stats_by_direction(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        stats = []
        for direct in ['Compra', 'Venda']:
            d = df[df['direction']==direct]
            if len(d) > 0:
                stats.append({
                    'direction': direct, 
                    'trades': len(d), 
                    'win_rate': float(round((d['winner'].sum()/len(d)*100), 2)), 
                    'total_profit': float(round(d['result'].sum(), 2)),
                    'avg_profit': float(round(d['result'].mean(), 2)),
                    'sharpe_ratio': float(round(self.calculate_sharpe(d['result']), 2))
                })
            else:
                stats.append({
                    'direction': direct, 'trades': 0, 'win_rate': 0, 'total_profit': 0, 'avg_profit': 0, 'sharpe_ratio': 0
                })
        return stats

    def get_recent_trades(self, n=100, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        return df.tail(n)[['entry_time', 'entry_price', 'exit_price', 'result', 'winner', 'direction', 'slope_degrees']].astype(str).to_dict('records') # Cast str for safety

    def get_equity_curve(self, use_filtered=False):
        df = self.filtered_df if use_filtered else self.trades_df
        if df.empty: return []
        # Downsample to max ~1000 points to prevent browser freeze with 170k+ trades
        step = max(1, len(df) // 1000)
        return df.iloc[::step][['entry_time', 'cumulative_result']].astype(str).to_dict('records')

    def get_available_filters(self):
        if self.trades_df.empty: return {}
        return {
            'hours': sorted(self.trades_df['hour'].unique().tolist()),
            'weekdays': sorted(self.trades_df['weekday'].unique().tolist()),
            'years': sorted(self.trades_df['year'].unique().tolist()),
            'volatility_levels': ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'],
            'dist_levels': [f'D{i}' for i in range(1, 11)],
            'di_levels': [f'DI{i}' for i in range(1, 11)],
            'angle_ranges': [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+'],
            'acc_levels': [f'ACC{i}' for i in range(1, 21)],
            'vol_slope_levels': ['Queda Forte', 'Queda Leve', 'Flat', 'Subida Leve', 'Subida Forte'],
            'jerk_levels': [f'J{i}' for i in range(1, 21)]
        }
