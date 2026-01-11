import pandas as pd
import numpy as np

# CONFIG
DATA_FILE = "attached_assets/win_1767085916180.txt"
FUTURE_LOOKAHEAD = 100 

def load_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    # Indicators
    df['sma20'] = df['close'].rolling(window=20).mean()
    return df.dropna().reset_index(drop=True)

def check_base_stats():
    df = load_data(DATA_FILE)
    
    total_signals = 0
    wins = 0
    losses = 0
    
    c = df['close'].values
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    sma = df['sma20'].values
    
    print(f"Scanning {len(df)} candles for Dynamic Range Stats...")
    
    for i in range(50, len(df)-FUTURE_LOOKAHEAD):
        if i % 200000 == 0: print(f"Index {i}...")
        
        # BUY: i-2 Bull, i-1 Bear, SMA Up, Low < Low(i-3)
        # i-2 index in array is i-2. 
        is_buy = (c[i-2] > o[i-2]) and (c[i-1] < o[i-1]) and (c[i-2] > sma[i-2]) and (l[i-1] < l[i-3])
        
        # SELL: i-2 Bear, i-1 Bull, SMA Down, High > High(i-3)
        is_sell = (c[i-2] < o[i-2]) and (c[i-1] > o[i-1]) and (c[i-2] < sma[i-2]) and (h[i-1] > h[i-3])
        
        if is_buy:
            total_signals += 1
            rng = h[i-1] - l[i-1]
            if rng == 0: rng = 5
            
            # Dynamic: Target = Range, Stop = Range
            target = c[i-1] + rng
            stop = c[i-1] - rng
            
            outcome = 0
            for k in range(i, i+FUTURE_LOOKAHEAD): 
                if h[k] >= target:
                    outcome = 1; break
                if l[k] <= stop:
                    outcome = 0; break
            if outcome == 1: wins += 1
            else: losses += 1
            
        elif is_sell:
            total_signals += 1
            rng = h[i-1] - l[i-1]
            if rng == 0: rng = 5
            
            target = c[i-1] - rng
            stop = c[i-1] + rng
            
            outcome = 0
            for k in range(i, i+FUTURE_LOOKAHEAD):
                if l[k] <= target:
                    outcome = 1; break
                if h[k] >= stop:
                    outcome = 0; break
            if outcome == 1: wins += 1
            else: losses += 1

    print("-" * 30)
    print(f"Total Signals: {total_signals}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    rate = wins / total_signals if total_signals > 0 else 0
    print(f"Base Win Rate (Dynamic 1:1): {rate:.2%}")

if __name__ == "__main__":
    check_base_stats()
