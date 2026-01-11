
import pandas as pd
import numpy as np
import torch
import os
from strategy_analyzer import StrategyAnalyzer

# --- Configuration ---
WINDOW_SIZE = 11  # From Paper (11 candles history)
NUM_FEATURES = 5  # SCP, RSI, WR, MFI, MACD
TARGET_CLASS_COUNT = 5000 # Balanced classes
FUTURE_LOOKAHEAD = 15 # For labeling local extrema

def calculate_technical_indicators(df):
    """Calculates the 5 specific features for MCNN-TC."""
    
    # 1. SCP (Stationary Closing Price) -> Log Return or Pct Change
    # Paper uses normalized differences. Tanh normalization happens at extraction.
    df['scp'] = df['close'].pct_change()
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Williams %R (14)
    # (Highest High - Close) / (Highest High - Lowest Low) * -100
    r_window = 14
    hh = df['high'].rolling(r_window).max()
    ll = df['low'].rolling(r_window).min()
    df['wr'] = -100 * ((hh - df['close']) / (hh - ll + 1e-9))
    
    # 4. MFI (Money Flow Index) (14)
    # Typical Price = (H+L+C)/3
    # Raw Money Flow = TP * Vol
    # Ratio = Pos Flow / Neg Flow
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    rmf = tp * df['tick_volume'] # Use tick_volume if real missing
    
    # Pos/Neg Flow
    tp_prev = tp.shift(1)
    pos_flow = pd.Series(np.where(tp > tp_prev, rmf, 0), index=df.index)
    neg_flow = pd.Series(np.where(tp < tp_prev, rmf, 0), index=df.index)
    
    pos_mf = pos_flow.rolling(14).sum()
    neg_mf = neg_flow.rolling(14).sum()
    
    mfi_ratio = pos_mf / (neg_mf + 1e-9)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    # 5. MACD (12, 26, 9) - We use the Histogram? Or Signal? Paper says "MACD".
    # Usually MACD Line and Signal Line. Let's use Normalized MACD Line.
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    
    return df

def get_labels_local_extrema(df, lookahead=10):
    """
    Labels:
    0: Hold (Noise)
    1: Buy (Local Minimum)
    2: Sell (Local Maximum)
    
    Logic: A point i is a Local Min if Low[i] is the minimum in [i-L, i+L].
    """
    labels = np.zeros(len(df), dtype=int)
    
    # Use 'close' or 'low'/'high'? Paper usually uses Close for simplicity or High/Low for precision.
    # Let's use Low/High for turning points.
    lows = df['low'].values
    highs = df['high'].values
    
    # We scan with a window
    # Actually, to train, we check if today (i) was a bottom/top confirmed by future data.
    
    for i in range(lookahead, len(df) - lookahead):
        # Local Min (Buy)
        # Check current Low vs surroundings
        window_lows = lows[i-lookahead : i+lookahead+1]
        if lows[i] == np.min(window_lows):
            labels[i] = 1 # Buy
            
        # Local Max (Sell)
        # Check current High vs surroundings
        window_highs = highs[i-lookahead : i+lookahead+1]
        if highs[i] == np.max(window_highs):
            labels[i] = 2 # Sell
            
    return labels

def normalize(x, method='minmax', range_min=0, range_max=1):
    if method == 'tanh':
        return np.tanh(x)
    return x # Placeholder, we do scaling per window

def generate_mcnn_dataset():
    data_file = "attached_assets/win_1767085916180.txt"
    if not os.path.exists(data_file):
        print("Data file not found.")
        return

    print("Loading Data...")
    df = pd.read_csv(data_file)
    
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    # Expect: time, open, high, low, close, tick_volume...
    
    # 1. Calc Features
    print("Calculating Indicators (MCNN)...")
    df = calculate_technical_indicators(df)
    
    # 2. Labeling (Turning Points)
    print("Labeling Local Extrema (Ground Truth)...")
    lookahead = 15 # H=15 candle window radius for min/max
    labels_all = get_labels_local_extrema(df, lookahead=lookahead)
    
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    # Re-align labels (naive, we lost ~30 rows at start)
    # Better: assign labels to df then dropna
    df['label'] = pd.Series(labels_all) # Align by index
    df = df.dropna().reset_index(drop=True)
    
    # 3. Extract Windows
    print("Extracting Windows (5x11 Images)...")
    
    # Containers
    X_buy, X_sell, X_hold = [], [], []
    
    # Features of interest
    feat_cols = ['scp', 'rsi', 'wr', 'mfi', 'macd']
    # Normalize globally or locally? 
    # MCNN paper typically uses specific normalization.
    # SCP -> Tanh (already centered around 0).
    # RSI, WR, MFI -> 0-100. Scale to 0-1.
    # MACD -> Unbounded. Scale ??
    
    # Global Pre-scaling
    df['scp'] = np.tanh(df['scp'] * 100) # Scale up small returns before tanh
    df['rsi'] = df['rsi'] / 100.0
    df['wr'] = (df['wr'] + 100) / 100.0 # -100..0 -> 0..1
    df['mfi'] = df['mfi'] / 100.0
    # MACD: Let's simpler robust scale
    macd_mean = df['macd'].mean()
    macd_std = df['macd'].std()
    df['macd'] = (df['macd'] - macd_mean) / (macd_std + 1e-9)
    df['macd'] = np.clip(df['macd'], -3, 3) / 3.0 # -1 to 1 approx
    
    data_matrix = df[feat_cols].values # (N, 5)
    labels = df['label'].values
    
    # Collect indices for each class
    # We want balanced dataset
    
    idxs_buy = np.where(labels == 1)[0]
    idxs_sell = np.where(labels == 2)[0]
    idxs_hold = np.where(labels == 0)[0]
    
    # Filter for valid windows
    min_idx = WINDOW_SIZE
    idxs_buy = idxs_buy[idxs_buy >= min_idx]
    idxs_sell = idxs_sell[idxs_sell >= min_idx]
    idxs_hold = idxs_hold[idxs_hold >= min_idx]
    
    print(f"Candidates: Buy={len(idxs_buy)}, Sell={len(idxs_sell)}, Hold={len(idxs_hold)}")
    
    # Random Sample to Balance
    n_samples = min(TARGET_CLASS_COUNT, len(idxs_buy), len(idxs_sell))
    np.random.shuffle(idxs_buy)
    np.random.shuffle(idxs_sell)
    np.random.shuffle(idxs_hold)
    
    use_buy = idxs_buy[:n_samples]
    use_sell = idxs_sell[:n_samples]
    use_hold = idxs_hold[:n_samples]
    
    final_indices = np.concatenate([use_buy, use_sell, use_hold])
    final_labels = np.concatenate([
        np.ones(len(use_buy)), 
        np.ones(len(use_sell)) * 2, 
        np.zeros(len(use_hold))
    ])
    
    # Extract
    X_data = []
    
    for i in final_indices:
        # Window: [i-10 ... i] (11 candles)
        # Shape: (11, 5) -> Transpose to (5, 11) for Conv2D (Height=Features, Width=Time)
        window = data_matrix[i-WINDOW_SIZE+1 : i+1]
        
        # Verify shape
        if window.shape != (WINDOW_SIZE, NUM_FEATURES):
            continue
            
        # Transpose: Row=Feature, Col=Time
        window_t = window.T # (5, 11)
        X_data.append(window_t)
        
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(final_labels, dtype=np.longlong)
    
    # Add Channel Dim: (N, 1, 5, 11)
    X_data = np.expand_dims(X_data, axis=1)
    
    print(f"Dataset Shape: {X_data.shape}, Labels: {y_data.shape}")
    
    torch.save(torch.from_numpy(X_data), 'train_inputs_mcnn.pt')
    torch.save(torch.from_numpy(y_data), 'train_labels_mcnn.pt')
    print("Saved train_inputs_mcnn.pt and train_labels_mcnn.pt")

if __name__ == "__main__":
    generate_mcnn_dataset()
