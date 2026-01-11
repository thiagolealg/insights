import pandas as pd
import numpy as np
import math
from datetime import datetime, time
import pickle
import os
import hashlib
import torch
import torch.nn as nn

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
    
    def __init__(self, data_file, target_points=50, stop_points=50, strategy_type='inside_bar'):
        self.data_file = data_file
        self.target_points = target_points
        self.stop_points = stop_points
        self.strategy_type = strategy_type
        self.df = None
        self.signals = []
        self.all_trades_df = pd.DataFrame() 
        self.current_ratio = '1:2'
        self.trades = []
        self.trades_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self._filtered_df = None
        
        # AI Init
        self.ai_model = None
        self.ai_device = torch.device('cpu')
        self.load_ai_model()

    def load_ai_model(self):
        try:
            model_path = "trade_mlp.pth"
            if os.path.exists(model_path):
                self.ai_model = TradeMLP().to(self.ai_device)
                self.ai_model.load_state_dict(torch.load(model_path, map_location=self.ai_device, weights_only=False))
                self.ai_model.eval()
                print("StrategyAnalyzer: AI Model loaded successfully.")
            else:
                print("StrategyAnalyzer: AI Model not found. Running without AI filter.")
        except Exception as e:
            print(f"StrategyAnalyzer: Error loading AI model: {e}")
            self.ai_model = None
        
    def _get_cache_filename(self):
        """Gera um nome de arquivo único para o cache baseado no arquivo de dados e na estratégia"""
        if not os.path.exists('.cache'):
            os.makedirs('.cache', exist_ok=True)

        
        hasher = hashlib.md5()
        with open(self.data_file, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        file_hash = hasher.hexdigest()
        
        return f".cache/strategy_{self.strategy_type}_{file_hash}_v20.pkl"

    def load_cache(self):
        """Tenta carregar resultados do cache"""
        cache_file = self._get_cache_filename()
        if os.path.exists(cache_file):
            try:
                print(f"Loading cache from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    self.all_trades_df = pickle.load(f)
                return not self.all_trades_df.empty
            except Exception as e:
                print(f"Error loading cache: {e}")
                return False
        return False

    def save_cache(self):
        """Salva resultados no cache"""
        try:
            cache_file = self._get_cache_filename()
            print(f"Saving cache to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.all_trades_df, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_data(self):
        # Try Parquet first (10x faster), fallback to CSV and convert
        parquet_path = self.data_file.replace('.txt', '.parquet').replace('.csv', '.parquet')
        
        if os.path.exists(parquet_path):
            self.df = pd.read_parquet(parquet_path)
            print(f"Loaded from Parquet (fast): {len(self.df)} rows")
        else:
            print("Parquet not found, loading from CSV and converting...")
            self.df = pd.read_csv(self.data_file)
            self.df['time'] = pd.to_datetime(self.df['time'])
            # Save as Parquet for future fast loading
            try:
                self.df.to_parquet(parquet_path, index=False)
                print(f"Converted to Parquet: {parquet_path}")
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

        # --- AI Features Pre-Calculation ---
        # 1. Slope
        self.df['sma_slope'] = self.df['sma20'].diff()
        
        # 2. RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. Volatility
        self.df['std_dev'] = self.df['close'].rolling(window=20).std()
        self.df['vol_long'] = self.df['std_dev'].rolling(window=100).mean()
        self.df['vol_norm'] = self.df['std_dev'] / (self.df['vol_long'] + 1e-9)

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
        
        # Cleanup temp column
        self.df.drop(columns=['close_rounded'], inplace=True, errors='ignore')
        
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

    def check_ai_filter(self, i, is_buy):
        if self.ai_model is None: return True # Bypass if no model
        
        # Safe access
        if i >= len(self.df): return False
        
        try:
            row = self.df.iloc[i]
            
            rsi = row['rsi']
            slope = row['sma_slope']
            vol = row['vol_norm']
            dist = row['dist_sma']
            body = row['body_pct']
            wick_u = row['wick_u_pct']
            wick_l = row['wick_l_pct']
            
            # Check NaNs
            if pd.isna(rsi) or pd.isna(slope): return False
            
            if not is_buy: # Invert for Sell
                rsi = 100.0 - rsi
                slope = -slope
                dist = -dist
                # Swap Wicks
                tmp = wick_u
                wick_u = wick_l
                wick_l = tmp
                
            # Normalize
            f_rsi = (rsi - 50.0) / 50.0
            f_slope = np.clip(slope / 5.0, -1.0, 1.0)
            f_vol = np.clip(vol - 1.0, -1.0, 1.0)
            f_dist = np.clip(dist / 5.0, -1.0, 1.0)
            f_body = (body - 0.5) * 2.0
            f_w_u = (wick_u - 0.5) * 2.0
            f_w_l = (wick_l - 0.5) * 2.0
            
            features = np.array([f_rsi, f_slope, f_vol, f_dist, f_body, f_w_u, f_w_l], dtype=np.float32)
            
            # Predict
            with torch.no_grad():
                t_x = torch.from_numpy(features).unsqueeze(0).to(self.ai_device) # (1, 7)
                out = self.ai_model(t_x)
                probs = torch.softmax(out, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # Threshold 0.60
                if pred.item() == 1 and conf.item() >= 0.60:
                    return True
                return False
                
        except Exception as e:
            # print(f"AI Check Error: {e}")
            return False

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
        
        for j in range(entry_idx, len(self.df)):
            exit_bars += 1
            if headers[j] >= target_price:
                exit_price = target_price
                winner = True
                break
            if lowers[j] <= stop_price:
                exit_price = stop_price
                winner = False
                break
            exit_price = closes[j] # Mark to market se acabar dataframe
            
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
            'distance_index_i2': float(self.df['distance_index'].values[entry_idx - 2]) if not pd.isna(self.df['distance_index'].values[entry_idx - 2]) else 0.0
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
        
        for j in range(entry_idx, len(self.df)):
            exit_bars += 1
            if lowers[j] <= target_price:
                exit_price = target_price
                winner = True
                break
            if headers[j] >= stop_price:
                exit_price = stop_price
                winner = False
                break
            exit_price = closes[j]
            
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
            'distance_index_i2': float(self.df['distance_index'].values[entry_idx - 2]) if not pd.isna(self.df['distance_index'].values[entry_idx - 2]) else 0.0
        }

    def run_backtest(self):
        # Tenta carregar do cache antes de carregar dados pesados
        # Mas para verificar hash, precisa ler o arquivo.
        # load_cache chama _get_cache_filename que lê o arquivo.
        
        if self.load_cache():
            self.set_active_ratio(self.AVAILABLE_RATIOS[2]['label'])
            print("Loaded from Cache.")
            return

        self.load_data() # Load heavy data only if needed

        self.signals = []
        i = 20 # Start check index
        length = len(self.df)
        
        print("Finding signals...")
        # Loop otimizado? Não, Python puro.
        while i < length - 1:
            is_buy = False
            is_sell = False
            
            if self.strategy_type == 'micro_channel':
                is_buy = self.check_buy_conditions_micro(i)
                is_sell = self.check_sell_conditions_micro(i)
            else:
                is_buy = self.check_buy_conditions(i)
                is_sell = self.check_sell_conditions(i)
            
            if is_buy:
                self.signals.append({'idx': i, 'direction': 'buy'})
                i += 1 # Check next candle? Usually skip inside trades?
            elif is_sell:
                self.signals.append({'idx': i, 'direction': 'sell'})
                i += 1
            else:
                i += 1
        
        print(f"Found {len(self.signals)} signals. Calculating all ratios...")
        
        all_trades = []
        for ratio in self.AVAILABLE_RATIOS:
            tp_mult = ratio['tp']
            sl_mult = ratio['sl']
            ratio_label = ratio['label']
            
            for signal in self.signals:
                if signal['direction'] == 'buy':
                    trade = self.simulate_buy_trade(signal['idx'], tp_mult, sl_mult, ratio_label)
                else:
                    trade = self.simulate_sell_trade(signal['idx'], tp_mult, sl_mult, ratio_label)
                all_trades.append(trade)
        
        self.all_trades_df = pd.DataFrame(all_trades)
        
        # Calculate Volatility Levels globally
        if not self.all_trades_df.empty:
            try:
                self.all_trades_df['vol_level'] = pd.qcut(self.all_trades_df['volatility'], q=5, labels=['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta'], duplicates='drop').astype(str)
            except: self.all_trades_df['vol_level'] = 'Média'
            
            try:
                self.all_trades_df['dist_level'] = pd.qcut(self.all_trades_df['dist_sma_i2'], q=10, labels=[f'D{i}' for i in range(1, 11)], duplicates='drop').astype(str)
            except: self.all_trades_df['dist_level'] = 'D5'
            
            try:
                self.all_trades_df['di_level'] = pd.qcut(self.all_trades_df['distance_index_i2'].abs(), q=10, labels=[f'DI{i}' for i in range(1, 11)], duplicates='drop').astype(str)
            except: self.all_trades_df['di_level'] = 'DI5'

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
            'angle_ranges': [f'{i}-{i+5}°' for i in range(0, 90, 5)] + ['90°+'] 
        }
