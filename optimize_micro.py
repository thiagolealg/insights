import pandas as pd
import numpy as np

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
    
    return df

def simulate_trade(df, entry_idx, direction):
    entry_price = df['close'].iloc[entry_idx]
    
    # Standard Signal Range (i)
    signal_high = df['high'].iloc[entry_idx]
    signal_low = df['low'].iloc[entry_idx]
    signal_range = signal_high - signal_low
    if signal_range <= 0: signal_range = 5 
    
    # Standard Ratio 1:2
    tp_ratio = 1.0
    sl_ratio = 2.0
    
    if direction == 'buy':
        target = entry_price + (signal_range * tp_ratio)
        stop = entry_price - (signal_range * sl_ratio)
    else:
        target = entry_price - (signal_range * tp_ratio)
        stop = entry_price + (signal_range * sl_ratio)
        
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
        'Current_Micro': {'wins': 0, 'losses': 0, 'result': 0.0},
        'User_Literal_RisingHighs': {'wins': 0, 'losses': 0, 'result': 0.0},
        'Corrected_FallingHighs': {'wins': 0, 'losses': 0, 'result': 0.0},
    }
    
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    smas = df['sma20'].values
    
    # Limit for speed
    if len(df) > 200000:
        print("Subsetting last 200k candles...")
        start_idx = len(df) - 200000
    else:
        start_idx = 20
        
    print(f"Scanning from index {start_idx}...")
    
    for i in range(start_idx, len(df)-1):
        if np.isnan(smas[i-1]): continue
        
        # Micro Channel BUY Logic
        # Common: Trend UP
        is_uptrend = closes[i-1] > smas[i-1] and smas[i-1] > smas[i-2]
        
        # 1. Current: 3 Lower Lows (L[i-3] > L[i-2] > L[i-1])
        # i is current, building. Signals are checked on completed candles?
        # Usually i-1 is signal.
        # "Low[i-3] > Low[i-2] > Low[i-1]" implies signals at i-1 close.
        curr_buy = is_uptrend and (lows[i-3] > lows[i-2]) and (lows[i-2] > lows[i-1])
        
        # 2. User Literal: 3 Rising Highs (H[i-4] < H[i-3] < H[i-2]) + Liquidity Grab (L[i-1] < L[i-4])
        # Requires i-4.
        if i < 5: continue
        
        user_rising = (highs[i-4] < highs[i-3]) and (highs[i-3] < highs[i-2])
        user_grab = lows[i-1] < lows[i-4]
        
        user_lit_buy = is_uptrend and user_rising and user_grab
        
        # 3. Corrected?: 3 Falling Highs (H[i-4] > H[i-3] > H[i-2]) + Liquidity Grab (L[i-1] < L[i-4])
        # This is a Bear Channel + Deep Flush.
        corr_falling = (highs[i-4] > highs[i-3]) and (highs[i-3] > highs[i-2])
        corr_grab = lows[i-1] < lows[i-4]
        
        corr_buy = is_uptrend and corr_falling and corr_grab
        
        
        # EXECTUE
        if curr_buy:
            res = simulate_trade(df, i-1, 'buy')
            if res > 0: variations['Current_Micro']['wins'] += 1
            else: variations['Current_Micro']['losses'] += 1
            variations['Current_Micro']['result'] += res
            
        if user_lit_buy:
            res = simulate_trade(df, i-1, 'buy')
            if res > 0: variations['User_Literal_RisingHighs']['wins'] += 1
            else: variations['User_Literal_RisingHighs']['losses'] += 1
            variations['User_Literal_RisingHighs']['result'] += res
            
        if corr_buy:
            res = simulate_trade(df, i-1, 'buy')
            if res > 0: variations['Corrected_FallingHighs']['wins'] += 1
            else: variations['Corrected_FallingHighs']['losses'] += 1
            variations['Corrected_FallingHighs']['result'] += res

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
