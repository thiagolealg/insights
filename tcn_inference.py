
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from generate_tcn_dataset import calculate_tcn_features
from train_tcn import TradeTCN
import os
from sklearn.preprocessing import StandardScaler

def run_tcn_inference():
    print("--- TCN Inference (Validation) ---")
    
    # Re-instantiate Model
    device = torch.device("cpu") # Inference on CPU is fine
    # Architecture Params from Training
    INPUT_CHANNELS = 9
    NUM_CLASSES = 3
    CHANNEL_SIZES = [64, 64, 64, 64]
    KERNEL_SIZE = 3
    DROPOUT = 0.0 # No dropout in eval
    
    model = TradeTCN(INPUT_CHANNELS, NUM_CLASSES, CHANNEL_SIZES, KERNEL_SIZE, DROPOUT).to(device)
    try:
        model.load_state_dict(torch.load("trade_tcn.pth", map_location=device, weights_only=False))
    except:
        print("Model file not found.")
        return
        
    model.eval()
    print("TCN Model Loaded.")
    
    # Load Data
    data_file = "attached_assets/win_1767085916180.txt"
    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]
    
    feature_df, feature_cols = calculate_tcn_features(df)
    
    # Scaling
    # We fit on the whole dataset to approximate the training scaling
    scaler = StandardScaler()
    data_vals = scaler.fit_transform(feature_df[feature_cols].values)
    
    # TCN requires scaling to be correct.
    
    # Inference Loop
    print("Running Sliding Window Inference...")
    WINDOW_SIZE = 120
    BATCH_SIZE = 1000
    
    predictions = []
    confidences = []
    indices = []
    
    # We can perform batch inference
    # Input: (Batch, 9, 120)
    
    total_len = len(feature_df)
    
    # Pre-calculate windows? Too much memory.
    # Just iterate.
    
    highs = feature_df['high'].values
    lows = feature_df['low'].values
    closes = feature_df['close'].values
    
    # Optimization: Filter candidates first?
    # We only care about predicting when an Inside Bar occurs.
    # If the strategy is "Inside Bar Breakout", we don't need to predict at every candle.
    # We check: Is 'i' an Inside Bar? If yes, ask AI "Will it break up/down?".
    
    print("Filtering candidates (Inside Bars only)...")
    candidate_indices = []
    for i in range(WINDOW_SIZE, total_len):
        # Inside Bar Check
        # High[i] <= High[i-1] and Low[i] >= Low[i-1]
        if (highs[i] <= highs[i-1]) and (lows[i] >= lows[i-1]):
            candidate_indices.append(i)
            
    print(f"Candidates (Inside Bars): {len(candidate_indices)}")
    
    # Process Candidates in Batches
    batch_X = []
    batch_idx = []
    
    for idx in candidate_indices:
        # Extract Window i-119 to i
        window = data_vals[idx-WINDOW_SIZE+1 : idx+1]
        if window.shape != (WINDOW_SIZE, 9): continue
        
        # Transpose to (9, 120)
        batch_X.append(window.T)
        batch_idx.append(idx)
        
        if len(batch_X) >= BATCH_SIZE:
            # Predict
            X_tensor = torch.tensor(np.array(batch_X), dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(X_tensor)
                probs = torch.softmax(out, dim=1)
                confs, preds = torch.max(probs, 1)
                
                predictions.extend(preds.numpy())
                confidences.extend(confs.numpy())
                indices.extend(batch_idx) # Which candles
            
            batch_X = []
            batch_idx = []
            
    # Final Batch
    if batch_X:
        X_tensor = torch.tensor(np.array(batch_X), dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(X_tensor)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, 1)
            predictions.extend(preds.numpy())
            confidences.extend(confs.numpy())
            indices.extend(batch_idx)
            
    # Analysis
    indices = np.array(indices)
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Threshold
    THRESHOLD = 0.60 # Goal
    
    buy_mask = (predictions == 1) & (confidences >= THRESHOLD)
    sell_mask = (predictions == 2) & (confidences >= THRESHOLD)
    
    buys = indices[buy_mask]
    sells = indices[sell_mask]
    
    print(f"Algorithm Selected: {len(buys)} Buys, {len(sells)} Sells")
    
    # Calc PnL
    win_b = 0
    tot_b = 0
    pnls = []
    
    LOOKAHEAD = 10 # Matches training labeling
    
    for idx in buys:
        if idx + LOOKAHEAD >= total_len: continue
        entry = closes[idx]
        # Max high in next 10? Or Close at 10?
        # Simple test: Close at 10 vs Entry
        exit_p = closes[idx+LOOKAHEAD]
        if exit_p > entry: win_b += 1
        tot_b += 1
        pnls.append(exit_p - entry)
        
    win_s = 0
    tot_s = 0
    for idx in sells:
        if idx + LOOKAHEAD >= total_len: continue
        entry = closes[idx]
        exit_p = closes[idx+LOOKAHEAD]
        if exit_p < entry: win_s += 1
        tot_s += 1
        pnls.append(entry - exit_p)
        
    total = tot_b + tot_s
    wins = win_b + win_s
    
    if total > 0:
        wr = 100.0 * wins / total
        print(f"--- TCN Results (Inside Bar Only) ---")
        print(f"Win Rate: {wr:.2f}%")
        print(f"Total Trades: {total}")
        print(f"Total PnL: {sum(pnls):.2f}")
    else:
        print("No trades selected.")

if __name__ == "__main__":
    run_tcn_inference()
