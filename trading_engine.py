"""
Trading Engine V1 - High Performance Strategy Executor
======================================================
Monitora arquivo de dados e executa +55 estratégias em tempo real.
Otimizado para rodar localmente sem Flask.

Uso:
    python trading_engine.py
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from strategy_analyzer import StrategyAnalyzer
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

DATA_FILE = "win_full_data.parquet"
STRATEGIES_FILE = "saved_strategies.json"
CHECK_INTERVAL_SEC = 1 # Fast loop, but effective delay is data reload time

class TradingEngine:
    def __init__(self):
        self.strategies = []
        self.last_data_mtime = 0
        self.shared_df = None
        self.analyzers = {} # Map 'id' -> StrategyAnalyzer instance
        
    def load_strategies_config(self):
        """Carrega configurações de estratégias salvas."""
        if not os.path.exists(STRATEGIES_FILE):
            print(f"{Fore.RED}Erro: Arquivo {STRATEGIES_FILE} não encontrado.{Style.RESET_ALL}")
            return []
        
        try:
            with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Erro ao ler JSON de estratégias: {e}{Style.RESET_ALL}")
            return []

    def check_data_update(self):
        """Verifica se o arquivo de dados mudou (nova vela)."""
        if not os.path.exists(DATA_FILE):
            return False
            
        try:
            stat = os.stat(DATA_FILE)
            if stat.st_mtime > self.last_data_mtime:
                self.last_data_mtime = stat.st_mtime
                return True
            return False
        except OSError:
            return False

    def reload_data(self):
        """Carrega dados do Parquet para memória compartilhada."""
        try:
            start_time = time.time()
            # Read parquet
            # Optimization: Read only necessary columns if possible, but full reload is safer for derived columns
            self.shared_df = pd.read_parquet(DATA_FILE)
            
            # Ensure datetime
            if not np.issubdtype(self.shared_df['time'].dtype, np.datetime64):
                self.shared_df['time'] = pd.to_datetime(self.shared_df['time'])
                
            # Pre-calculate common indicators globally to save time for 55 instances?
            # StrategyAnalyzer calculates indicators inside __init__ (load_data called by init logic?).
            # Wait, StrategyAnalyzer calculates indicators inside load_data.
            # If we pass DF, it assumes it's raw and calculates indicators inside?
            # Let's check StrategyAnalyzer logic.
            # load_data calls:
            #   if self.df is not None: return
            #   ... calculations ...
            # Wait! If we pass raw DF, the calculations (SMA, RSI, etc) are skipped if we return early!
            # We need to process the DF once globally.
            
            self._process_indicators(self.shared_df)
            
            elapsed = (time.time() - start_time) * 1000
            print(f"{Fore.CYAN}[DATA] Dados atualizados em {elapsed:.1f}ms (Candles: {len(self.shared_df)}){Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}[DATA] Erro ao carregar dados: {e}{Style.RESET_ALL}")
            return False

    def _process_indicators(self, df):
        """Calcula indicadores globais para o DataFrame compartilhado."""
        # Replication of StrategyAnalyzer.load_data logic
        if 'hour' not in df.columns: df['hour'] = df['time'].dt.hour
        if 'weekday' not in df.columns: df['weekday'] = df['time'].dt.dayofweek
        if 'year' not in df.columns: df['year'] = df['time'].dt.year
        if 'month' not in df.columns: df['month'] = df['time'].dt.month
        if 'year_month' not in df.columns: df['year_month'] = df['time'].dt.to_period('M').astype(str)
        
        # Ranges
        df['body_high'] = df[['open', 'close']].max(axis=1)
        df['body_low'] = df[['open', 'close']].min(axis=1)
        df['range'] = df['high'] - df['low']
        
        # SMA 20 (Critical)
        df['sma20'] = df['close'].rolling(window=20).mean()
        
        # ATR / Volatility
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(df['high'] - df['low'],
                           np.maximum((df['high'] - df['prev_close']).abs(),
                                      (df['low'] - df['prev_close']).abs()))
        df['atr'] = df['tr'].rolling(window=20).mean() # ATR 20
        df['std_dev'] = df['close'].rolling(window=20).std()
        
        # Body Pct / Wicks
        range_c = df['high'] - df['low']
        mask_valid = range_c > 0
        df['body_pct'] = 0.0
        df.loc[mask_valid, 'body_pct'] = (df.loc[mask_valid, 'close'] - df.loc[mask_valid, 'open']).abs() / range_c[mask_valid]
        
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Specific Strategies Helpers
        # STD Reversal
        std_candle = df[['high', 'low', 'close']].std(axis=1)
        cl_diff = df['close'] - df['low']
        hc_diff = df['high'] - df['close']
        
        # Calculate Rolling Sum manually or use pandas
        # Sell cond: std > cl_diff * 1.2
        sell_sig = (std_candle > (cl_diff * 1.2)).astype(int)
        df['rev_std_sell_signal'] = sell_sig.rolling(window=4).sum()
        
        buy_sig = (std_candle > (hc_diff * 1.2)).astype(int)
        df['rev_std_buy_signal'] = buy_sig.rolling(window=4).sum()
        
        # Distance Index (Vectorized version from strategy_analyzer)
        # We can reuse the logic code or just import/copy?
        # Ideally StrategyAnalyzer should have a static method, but let's replicate for speed here.
        # Minimal DI calculation:
        df['date'] = df['time'].dt.date
        df['close_rounded'] = df['close'].round(-1)
        importance_df = (df.groupby(['date', 'close_rounded'])
            .agg(count=('close', 'size'), vol_sum=('tick_volume', 'sum'))
            .reset_index())
        importance_df['importance'] = importance_df['count'] * importance_df['vol_sum']
        idx = importance_df.groupby('date')['importance'].idxmax()
        daily_important = importance_df.loc[idx].set_index('date')['close_rounded']
        daily_shifted = daily_important.shift(1)
        df['prev_day_important_price'] = df['date'].map(daily_shifted)
        
        denom = df['prev_day_important_price'].abs() + 1e-9
        df['distance_index'] = ((df['close'] - df['prev_day_important_price']) / denom) * 100
        
        # Delta T acceleration
        df['delta_di'] = df['distance_index'].diff()
        
        # Drop temp
        df.drop(columns=['close_rounded', 'date'], inplace=True, errors='ignore')


    def initialize_strategies(self):
        """Cria instâncias para cada estratégia salva usando o DF compartilhado."""
        configs = self.load_strategies_config()
        self.analyzers = {}
        
        print(f"Inicializando {len(configs)} estratégias...")
        
        for cfg in configs:
            s_type = cfg.get('strategy_type')
            s_name = cfg.get('name') or s_type
            s_id = str(cfg.get('id', hash(s_name))) # Use ID or generated hash
            
            # Create instance with SHARED DF (Lightweight)
            # Pass dummy data_file name since we inject DF
            analyzer = StrategyAnalyzer(DATA_FILE, strategy_type=s_type, data_frame=self.shared_df)
            
            # Store config with analyzer
            analyzer.config = cfg
            self.analyzers[s_id] = analyzer
            
        print(f"{Fore.GREEN}Motor pronto com {len(self.analyzers)} instâncias.{Style.RESET_ALL}")

    def run_loop(self):
        print(f"{Fore.YELLOW}=== TRADING ENGINE V1 INICIADO ==={Style.RESET_ALL}")
        print("Monitorando arquivo de dados a cada 1 seg...")
        print("Pressione Ctrl+C para parar.")
        
        # Initial Load
        if not self.reload_data():
            print("Aguardando arquivo de dados...")
        
        self.initialize_strategies()
        
        while True:
            try:
                # 1. Check for new data
                if self.check_data_update():
                    if self.reload_data():
                        # Update all instances with new DF pointer
                        # Since Python passes by reference, self.shared_df update might not propagate 
                        # if we replaced the object 'self.shared_df = pd.read...'.
                        # We need to update the analyzer.df reference manually.
                        for an in self.analyzers.values():
                            an.df = self.shared_df
                            
                        self.process_signals()
                
                # 2. Heartbeat / Status
                # Optional: Print "Waiting..." every 10s? No, keep clean.
                
                time.sleep(CHECK_INTERVAL_SEC)
                
            except KeyboardInterrupt:
                print("\nEncerrando Engine...")
                break
            except Exception as e:
                print(f"{Fore.RED}Erro no loop principal: {e}{Style.RESET_ALL}")
                time.sleep(5)

    def process_signals(self):
        """Verifica sinais para o ÚLTIMO candle fechado."""
        if self.shared_df is None or self.shared_df.empty:
            return

        last_idx = len(self.shared_df) - 1
        last_time = self.shared_df['time'].iloc[last_idx]
        last_close = self.shared_df['close'].iloc[last_idx]
        
        print(f"\n{Fore.WHITE}--- Processando Candle {last_time} (Close: {last_close}) ---{Style.RESET_ALL}")
        
        signals_found = 0
        start_proc = time.time()
        
        for s_id, analyzer in self.analyzers.items():
            cfg = analyzer.config
            name = cfg.get('name')
            s_type = cfg.get('strategy_type')
            
            # Check Buy
            # Note: check_buy_conditions checks entry at 'i'. 
            # If true, it means conditions at i-1, i-2 etc are met for entry at CLOSE(i).
            # So checking 'last_idx' implies checking if the JUST CLOSED candle triggers a buy.
            is_buy = analyzer.check_buy_conditions(last_idx)
            
            # Check Sell
            is_sell = analyzer.check_sell_conditions(last_idx)
            
            if is_buy:
                print(f"{Fore.GREEN}>> [COMPRA] {name} ({s_type}) @ {last_close}{Style.RESET_ALL}")
                signals_found += 1
                # TODO: Send order API / Telegram / Log to file
                
            if is_sell:
                print(f"{Fore.MAGENTA}>> [VENDA]  {name} ({s_type}) @ {last_close}{Style.RESET_ALL}")
                signals_found += 1

        elapsed = (time.time() - start_proc) * 1000
        if signals_found == 0:
            print(f"Nenhum sinal encontrado. (Processado em {elapsed:.1f}ms)")
        else:
            print(f"{Fore.YELLOW}Total de Sinais: {signals_found} (Processado em {elapsed:.1f}ms){Style.RESET_ALL}")
            # Beep or alert?
            print('\a') # Bell sound


if __name__ == "__main__":
    engine = TradingEngine()
    engine.run_loop()
