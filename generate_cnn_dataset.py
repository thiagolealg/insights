import pandas as pd
import numpy as np
import torch
import os
from strategy_analyzer import StrategyAnalyzer

# CONFIG
WINDOW_SIZE = 32 # Must match inference
FUTURE_LOOKAHEAD = 300 # 5 hours
TAKE_PROFIT = 100
STOP_LOSS = 200

# Select Strategy
# Select Strategy
STRATEGY_TYPE = 'bull_bear' # OPTIONS: 'bull_bear', 'inside_bar'

def extract_features_vector(df, i, invert=False):
    # MLP Feature Extraction (Trend/Price Action - Universal)
    # Features that worked for Bull-Bear:
    # 1. RSI
    # 2. SMA Slope
    # 3. Volatility (Norm)
    # 4. Dist SMA
    # 5. Body %
    # 6. Wick Upper %
    # 7. Wick Lower %
    
    if i < 40: return None
    row = df.iloc[i]
    
    rsi = row['rsi']
    slope = row['sma_slope']
    vol = row['vol_norm']
    dist = row['dist_sma']
    body = row['body_pct']
    wick_u = row['wick_u_pct']
    wick_l = row['wick_l_pct']
    
    if invert:
        # Invert Directional
        rsi = 100.0 - rsi
        slope = -slope
        dist = -dist
        # Swap Wicks (Upper becomes Lower in inverted view)
        tmp = wick_u
        wick_u = wick_l
        wick_l = tmp
        
    # Normalize
    f_rsi = (rsi - 50.0) / 50.0
    f_slope = np.clip(slope / 5.0, -1.0, 1.0)
    f_vol = np.clip(vol - 1.0, -1.0, 1.0)
    f_dist = np.clip(dist / 5.0, -1.0, 1.0)
    f_body = (body - 0.5) * 2.0
    f_wik_u = (wick_u - 0.5) * 2.0
    f_wik_l = (wick_l - 0.5) * 2.0
    
    features = np.array([f_rsi, f_slope, f_vol, f_dist, f_body, f_wik_u, f_wik_l], dtype=np.float32)
    return features

def check_outcome(df, i, is_buy):
    entry_price = df['close'].iloc[i]
    future = df.iloc[i+1 : i+FUTURE_LOOKAHEAD]
    if len(future) < 10: return 0 
    
    headers = future['high'].values
    lowers = future['low'].values
    
    if is_buy:
        tp = entry_price + TAKE_PROFIT
        sl = entry_price - STOP_LOSS
        for k in range(len(headers)):
            if lowers[k] <= sl: return 0 # Hit SL first
            if headers[k] >= tp: return 1 # Hit TP first
    else:
        tp = entry_price - TAKE_PROFIT
        sl = entry_price + STOP_LOSS
        for k in range(len(headers)):
            if headers[k] >= sl: return 0 # Hit SL first
            if lowers[k] <= tp: return 1 # Hit TP first
            
    return 0 # Timeout or no result

def generate_dataset():
    print(f"--- Generating Dataset for Strategy: {STRATEGY_TYPE} ---")
    data_file = "attached_assets/win_1767085916180.txt"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found")
        return

    # Load & Prep
    analyzer = StrategyAnalyzer(data_file, strategy_type=STRATEGY_TYPE)
    analyzer.load_data()
    df = analyzer.df
    print(f"Loaded {len(df)} candles")
    
    # Calc Features Global
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma_slope'] = df['sma20'].diff()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['std_dev'] = df['close'].rolling(window=20).std()
    vol_long = df['std_dev'].rolling(window=100).mean()
    df['vol_norm'] = df['std_dev'] / (vol_long + 1e-9)
    
    # Price Context
    df['dist_sma'] = (df['close'] - df['sma20']) / (df['sma20'] + 1e-9) * 1000.0 # Pts diff
    
    # Candle Shape
    candle_range = df['high'] - df['low']
    df['body_pct'] = (df['close'] - df['open']).abs() / (candle_range + 1e-9)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_u_pct'] = df['upper_wick'] / (candle_range + 1e-9)
    df['wick_l_pct'] = df['lower_wick'] / (candle_range + 1e-9)
    
    # CRITICAL FIX: Alignment
    # Analyzer uses self.df internally.
    # We must ensure self.df matches our feature-rich df AND index alignment.
    df = df.dropna().reset_index(drop=True)
    analyzer.df = df 
    print(f"Aligned DF: {len(df)} candles.")
    
    # Scan
    inputs = []
    labels = []
    
    count_win = 0
    count_loss = 0
    TARGET_PER_CLASS = 3000 
    
    start_index = WINDOW_SIZE + 100
    
    for i in range(start_index, len(df) - FUTURE_LOOKAHEAD):
        is_buy = False
        is_sell = False
        
        if STRATEGY_TYPE == 'bull_bear':
            is_buy = analyzer.check_buy_conditions_bull_bear(i)
            is_sell = analyzer.check_sell_conditions_bull_bear(i)
        elif STRATEGY_TYPE == 'inside_bar':
            is_buy = analyzer.check_buy_conditions_inside_bar(i)
            is_sell = analyzer.check_sell_conditions_inside_bar(i)
            
        if not (is_buy or is_sell): continue
        if is_buy and is_sell: continue

        res = check_outcome(df, i, is_buy)
        
        if res == 1:
            if count_win >= TARGET_PER_CLASS: continue
            count_win += 1
        else:
            if count_loss >= TARGET_PER_CLASS: continue
            count_loss += 1
            
        # MLP Vector
        feats = extract_features_vector(df, i, invert=is_sell)
        if feats is None: continue
        
        inputs.append(feats)
        labels.append(res)
        
        if len(inputs) % 1000 == 0:
            print(f"Collected {len(inputs)}... (W:{count_win} L:{count_loss})")
            
        if count_win >= TARGET_PER_CLASS and count_loss >= TARGET_PER_CLASS:
            break
            
    X = np.array(inputs, dtype=np.float32)
    y = np.array(labels, dtype=np.longlong)
    
    print(f"Inside Bar Dataset: {len(X)} samples. Wins: {count_win}, Losses: {count_loss}")
    
    torch.save(X, 'train_inputs.pt')
    torch.save(y, 'train_labels.pt')
    print("Dataset saved.")

if __name__ == "__main__":
    generate_dataset()
