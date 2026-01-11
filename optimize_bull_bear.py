import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
DATA_FILE = "attached_assets/win_1767085916180.txt"

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    
    # Calculate Indicators
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # ATR for normalization (optional)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
               np.maximum((df['high'] - df['prev_close']).abs(),
                          (df['low'] - df['prev_close']).abs()))
    df['atr'] = df['tr'].rolling(window=20).mean()
    
    return df

def simulate_trade(df, entry_idx, direction, tp_ratio=1.0, sl_ratio=2.0):
    entry_price = df['close'].iloc[entry_idx]
    
    # Standard Signal Candle Range (i) or Setup Range (i-1)?
    # Standard logic uses i-1 range usually.
    # Entry is at CLOSE of i. So Stop is based on i or i-1?
    # Usually: Stop below LOW of SIGNAL candle (i).
    
    signal_high = df['high'].iloc[entry_idx]
    signal_low = df['low'].iloc[entry_idx]
    signal_range = signal_high - signal_low
    if signal_range <= 0: signal_range = 5 # minimal
    
    if direction == 'buy':
        target = entry_price + (signal_range * tp_ratio)
        stop = entry_price - (signal_range * sl_ratio)
    else:
        target = entry_price - (signal_range * tp_ratio)
        stop = entry_price + (signal_range * sl_ratio)
        
    # Vectorized check? No, sequential for accuracy per trade
    # Limit max duration? 1000 bars
    subset = df.iloc[entry_idx+1:entry_idx+1001]
    
    for i, row in subset.iterrows():
        if direction == 'buy':
            if row['high'] >= target: return (target - entry_price)
            if row['low'] <= stop: return (stop - entry_price)
        else:
            if row['low'] <= target: return (entry_price - target)
            if row['high'] >= stop: return (entry_price - stop)
            
    # Time exit
    exit_price = subset.iloc[-1]['close']
    if direction == 'buy': return exit_price - entry_price
    else: return entry_price - exit_price

def run_strategies(df):
    variations = {
        'Base': {'wins': 0, 'losses': 0, 'result': 0.0},
        'Var1_Low_Break': {'wins': 0, 'losses': 0, 'result': 0.0},
        'Var2_Weak_Pullback': {'wins': 0, 'losses': 0, 'result': 0.0},
        'Var3_Higher_Low': {'wins': 0, 'losses': 0, 'result': 0.0},
    }
    
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    smas = df['sma20'].values
    bodies = df['body_size'].values
    
    print(f"Scanning {len(df)} candles...")
    
    # Limit to last 200k for speed
    if len(df) > 200000:
        df = df.iloc[-200000:].reset_index(drop=True)
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        smas = df['sma20'].values
        bodies = df['body_size'].values
        print(f"Subset to last 200,000 candles for performance.")
    
    for i in range(20, len(df)-1):
        if np.isnan(smas[i-2]): continue
        
        # Common Data Points
        c_i2, c_i1 = closes[i-2], closes[i-1] # i-2 is 'setup', i-1 is 'signal' candle? Be careful with indices.
        # User says: i-2 Alta, i-1 Baixa.
        # If we are at 'i', do we trade based on i-1 and i-2 closing? Yes.
        # Entry is at 'i' open (or i-1 close).
        
        o_i2, o_i1 = opens[i-2], opens[i-1]
        l_i3, l_i2, l_i1 = lows[i-3], lows[i-2], lows[i-1]
        
        # Conditions
        is_i2_bull = c_i2 > o_i2
        is_i1_bear = c_i1 < o_i1
        is_i2_bear = c_i2 < o_i2
        is_i1_bull = c_i1 > o_i1
        
        # BUY SCENARIOS
        # Base: i-2 Bull, i-1 Bear, i-2 > SMA
        base_buy = is_i2_bull and is_i1_bear and (c_i2 > smas[i-2])
        
        # Var 1: Base + Low[i-1] < Low[i-3] (Liquidity Grab / Stop Hunt)
        # Note: compare i-1 low vs i-3 low.
        var1_buy = base_buy and (l_i1 < l_i3)
        
        # Var 2: Base + Weak Pullback (Body[i-1] < Body[i-2] * 0.6)
        var2_buy = base_buy and (bodies[i-1] < bodies[i-2] * 0.6)
        
        # Var 3: Base + Higher Low (Structure: Low[i-1] > Low[i-2]) means pullback didn't break impulse start
        var3_buy = base_buy and (l_i1 > l_i2)
        
        # Execute Trades for matching variations
        # Note: multiple variations can trigger on same candle.
        
        # Simulate Result ONCE if any triggered
        active_vars = []
        if base_buy: active_vars.append('Base')
        if var1_buy: active_vars.append('Var1_Low_Break')
        if var2_buy: active_vars.append('Var2_Weak_Pullback')
        if var3_buy: active_vars.append('Var3_Higher_Low')
        
        if active_vars:
            res = simulate_trade(df, i-1, 'buy') # i-1 is the signal candle index
            is_win = res > 0
            for v in active_vars:
                if is_win: variations[v]['wins'] += 1
                else: variations[v]['losses'] += 1
                variations[v]['result'] += res

        # SELL SCENARIOS (Mirrored)
        base_sell = is_i2_bear and is_i1_bull and (c_i2 < smas[i-2])
        var1_sell = base_sell and (highs[i-1] > highs[i-3])
        var2_sell = base_sell and (bodies[i-1] < bodies[i-2] * 0.6)
        var3_sell = base_sell and (highs[i-1] < highs[i-2]) # Lower High
        
        active_vars_sell = []
        if base_sell: active_vars_sell.append('Base')
        if var1_sell: active_vars_sell.append('Var1_Low_Break')
        if var2_sell: active_vars_sell.append('Var2_Weak_Pullback')
        if var3_sell: active_vars_sell.append('Var3_Higher_Low')
        
        if active_vars_sell:
            res = simulate_trade(df, i-1, 'sell')
            is_win = res > 0
            for v in active_vars_sell:
                if is_win: variations[v]['wins'] += 1
                else: variations[v]['losses'] += 1
                variations[v]['result'] += res
                
    return variations

def print_results(variations):
    print("\n{:<25} | {:<10} | {:<10} | {:<10} | {:<15}".format("Variation", "Trades", "Win Rate", "Profit", "Avg Profit"))
    print("-" * 80)
    
    for name, stats in variations.items():
        total = stats['wins'] + stats['losses']
        if total == 0:
            print("{:<25} | {:<10} | {:<10} | {:<10} | {:<15}".format(name, 0, "0%", "0.0", "0.0"))
            continue
            
        win_rate = (stats['wins'] / total) * 100
        avg = stats['result'] / total
        print("{:<25} | {:<10} | {:<10.1f}% | {:<10.1f} | {:<15.2f}".format(
            name, total, win_rate, stats['result'], avg
        ))

if __name__ == "__main__":
    try:
        df = load_data(DATA_FILE)
        results = run_strategies(df)
        print_results(results)
    except Exception as e:
        print(f"Error: {e}")
