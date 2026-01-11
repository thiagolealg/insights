
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from strategy_analyzer import StrategyAnalyzer

# CONFIG
WINDOW_SIZE = 32 

class TradeMLP(nn.Module):
    def __init__(self):
        super(TradeMLP, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x

class AI_Predictor:
    def __init__(self, model_path="trade_mlp.pth"):
        self.device = torch.device("cpu") # MLP is fast on CPU
        self.model = TradeMLP().to(self.device)
        self.load_model(model_path)

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
            self.model.eval()
            print(f"AI Model loaded from {path}")
        else:
            print(f"Model file {path} not found! Train first.")
            self.model = None

def run_standalone_test():
    print("--- Running Standalone AI Test (Bull-Bear Trend - 60% Conf) ---")
    CONFIDENCE_THRESHOLD = 0.60 
    
    # Load Model
    predictor = AI_Predictor("trade_mlp.pth")
    if predictor.model is None: return

    data_file = "attached_assets/win_1767085916180.txt"
    if not os.path.exists(data_file): return

    analyzer = StrategyAnalyzer(data_file, strategy_type='bull_bear')
    analyzer.load_data()
    df = analyzer.df
    print(f"Loaded {len(df)} candles.")
    
    # --- 1. Vectorized Feature Engineering (Trend) ---
    print("Calculating Features (Trend / Price Action)...")
    
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. Slope
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma_slope'] = df['sma20'].diff()
    
    # 3. Volatility
    df['std_dev'] = df['close'].rolling(20).std()
    vol_long = df['std_dev'].rolling(100).mean()
    df['vol_norm'] = df['std_dev'] / (vol_long + 1e-9)
    
    # 4. Dist SMA
    df['dist_sma'] = (df['close'] - df['sma20']) / (df['sma20'] + 1e-9) * 1000.0
    
    # 5. Body %
    candle_range = df['high'] - df['low']
    df['body_pct'] = (df['close'] - df['open']).abs() / (candle_range + 1e-9)
    
    # 6/7. Wicks
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_u_pct'] = df['upper_wick'] / (candle_range + 1e-9)
    df['wick_l_pct'] = df['lower_wick'] / (candle_range + 1e-9)
    
    df = df.dropna().reset_index(drop=True)
    analyzer.df = df # Sync
    
    # Normalize (Vectorized)
    f_rsi = (df['rsi'].values - 50.0) / 50.0
    f_slope = np.clip(df['sma_slope'].values / 5.0, -1.0, 1.0)
    f_vol = np.clip(df['vol_norm'].values - 1.0, -1.0, 1.0)
    f_dist = np.clip(df['dist_sma'].values / 5.0, -1.0, 1.0)
    f_body = (df['body_pct'].values - 0.5) * 2.0
    f_w_u = (df['wick_u_pct'].values - 0.5) * 2.0
    f_w_l = (df['wick_l_pct'].values - 0.5) * 2.0
    
    # (N, 7)
    global_features = np.column_stack((f_rsi, f_slope, f_vol, f_dist, f_body, f_w_u, f_w_l)).astype(np.float32)
    
    # --- 2. Identify Signals ---
    FUTURE_LOOKAHEAD = 300
    scan_window = 200000
    start_index = max(WINDOW_SIZE+50, len(df) - scan_window)
    limit_idx = len(df) - FUTURE_LOOKAHEAD
    
    print(f"Scanning Signals in last {len(df)-start_index} candles...")
    
    signal_indices = []
    signal_types = [] 
    
    for i in range(start_index, limit_idx):
        if analyzer.check_buy_conditions_bull_bear(i):
            signal_types.append(1)
            signal_indices.append(i)
        elif analyzer.check_sell_conditions_bull_bear(i):
            signal_types.append(-1)
            signal_indices.append(i)
            
    signal_indices = np.array(signal_indices)
    signal_types = np.array(signal_types)
    num_signals = len(signal_indices)
    print(f"Found {num_signals} Signals.")
    
    if num_signals == 0: return

    # --- 3. Batch Inference ---
    BATCH_SIZE = 16384
    predictions = np.zeros(num_signals, dtype=int)
    confidences = np.zeros(num_signals, dtype=float)
    
    print(f"Running Inference...")
    predictor.model.eval()
    
    with torch.no_grad():
        for b_start in range(0, num_signals, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, num_signals)
            batch_idxs = signal_indices[b_start:b_end]
            batch_types = signal_types[b_start:b_end]
            
            batch_X = global_features[batch_idxs].copy()
            
            # Invert for Sells
            sell_mask = (batch_types == -1)
            if np.any(sell_mask):
                # RSI 
                batch_X[sell_mask, 0] = -batch_X[sell_mask, 0] 
                # Slope
                batch_X[sell_mask, 1] = -batch_X[sell_mask, 1]
                # Dist
                batch_X[sell_mask, 3] = -batch_X[sell_mask, 3]
                # Swap Wicks (U <-> L)
                tmp = batch_X[sell_mask, 5].copy()
                batch_X[sell_mask, 5] = batch_X[sell_mask, 6]
                batch_X[sell_mask, 6] = tmp
                
            t_X = torch.from_numpy(batch_X).to(predictor.device)
            out = predictor.model(t_X)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, 1)
            
            predictions[b_start:b_end] = preds.cpu().numpy()
            confidences[b_start:b_end] = confs.cpu().numpy()
            
    # --- 4. Calc Results ---
    strat_wins = 0; strat_losses = 0
    approved_wins = 0; approved_losses = 0
    ai_approved = 0
    
    print("Calculating Outcomes...")
    highs = df['high'].values
    lows = df['low'].values
    close_vals = df['close'].values
    
    # TP/SL logic same as generation (100/200?) Or 1:1?
    # Analyzer defaults: Target=50, Stop=50?
    # Let's check init: StrategyAnalyzer(data_file) uses default target=50.
    # But generate_cnn uses 100/200.
    # Let's use 100/100 for fairness of 1:1, or matching whatever generation did.
    # Generation used TP=100, SL=200? (See generate_cnn_dataset.py Config)
    # Yes: TAKE_PROFIT = 100, STOP_LOSS = 200.
    # So it trained on 1:2 R:R (Risk 200 to Make 100? No. Wait.)
    # TP 100 (Profit), SL 200 (Loss). That is negative R:R (Risk 2 to make 1).
    # That explains why baseline might be high WR (easier to hit 100 than 200).
    # Let's match it here.
    
    tp_dist = 100; sl_dist = 200
    
    for k in range(num_signals):
        idx = signal_indices[k]
        sType = signal_types[k]
        entry = close_vals[idx]
        
        real_result = 0
        f_h = highs[idx+1 : idx+FUTURE_LOOKAHEAD]
        f_l = lows[idx+1 : idx+FUTURE_LOOKAHEAD]
        if len(f_h) == 0: continue
        
        if sType == 1: # Buy
            tp = entry + tp_dist; sl = entry - sl_dist
            hit_tp = f_h >= tp
            hit_sl = f_l <= sl
            # Find first occurrence
            idx_tp = np.argmax(hit_tp) if np.any(hit_tp) else 9999
            idx_sl = np.argmax(hit_sl) if np.any(hit_sl) else 9999
            if idx_tp < idx_sl: real_result = 1
        else: # Sell
            tp = entry - tp_dist; sl = entry + sl_dist
            hit_tp = f_l <= tp
            hit_sl = f_h >= sl
            idx_tp = np.argmax(hit_tp) if np.any(hit_tp) else 9999
            idx_sl = np.argmax(hit_sl) if np.any(hit_sl) else 9999
            if idx_tp < idx_sl: real_result = 1
            
        if real_result == 1: strat_wins += 1
        else: strat_losses += 1
        
        # AI
        if predictions[k] == 1 and confidences[k] >= CONFIDENCE_THRESHOLD:
            ai_approved += 1
            if real_result == 1: approved_wins += 1
            else: approved_losses += 1
            
    print("\n" + "="*50)
    print(f"RESULTS (Bull-Bear @ {CONFIDENCE_THRESHOLD*100:.0f}% Confidence)")
    total = strat_wins + strat_losses
    strat_wr = strat_wins/total if total > 0 else 0
    print(f"Base Strategy Win Rate: {strat_wr:.2%} ({strat_wins}/{strat_losses})")
    print("-" * 50)
    
    app_wr = approved_wins/ai_approved if ai_approved > 0 else 0
    print(f"AI APPROVED: {ai_approved} Trades")
    print(f"AI Win Rate: {app_wr:.2%} ({approved_wins}/{approved_losses})")
    
    if strat_wr > 0:
        delta = app_wr - strat_wr
        print(f"AI Improvement: {delta:+.2%}")
    print("="*50)

if __name__ == "__main__":
    run_standalone_test()
