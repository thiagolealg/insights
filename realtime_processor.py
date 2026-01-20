import time
import pandas as pd
import numpy as np
import os
import sys
import json
import re
import gc
# import psutil # Optional
from colorama import init, Fore, Style
from datetime import datetime

# Try fetching MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print(Fore.RED + "MetaTrader5 module not found. Install with: pip install MetaTrader5" + Style.RESET_ALL)

# Initialize Colorama
init()

# Configuration
DATA_FILE = "win_full_data.parquet"
STRATEGIES_FILE = "saved_strategies.json"
ASSET_COUNT = 1 # Single Asset
HISTORY_SIZE = 100 
SMA_PERIOD = 20
MAX_LOSS_POINTS = 1000

class ActiveTrade:
    __slots__ = ['symbol', 'direction', 'entry_price', 'tp_price', 'sl_price', 'strategy_id', 'strategy_name', 'entry_time', 'ticket']
    def __init__(self, symbol, direction, entry_price, tp_price, sl_price, strategy_id, strategy_name, entry_time, ticket=0):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.entry_time = entry_time
        self.ticket = ticket

class StrategyManager:
    def __init__(self):
        self.strategies = []
        self.bins = {}
        self.daily_important_price = 0.0
        self.mock_mode = False # Shared Limit

        
    def load_and_prepare(self):
        print(f"{Fore.CYAN}Loading Configuration...{Style.RESET_ALL}")
        try:
            df = pd.read_parquet(DATA_FILE)
            self._calculate_bins(df)
            self.daily_important_price = df['close'].iloc[-1] 
        except:
            print(f"{Fore.RED}Data load failed (hist), using defaults.{Style.RESET_ALL}")
            self.bins = {'volatility': [0, 100], 'dist_levels': [0, 100], 'di_levels': [0, 100], 'acc_levels': [-100, 100]}
            
        try:
            with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
                raw_strategies = json.load(f)
            self.strategies = self._optimize_strategies(raw_strategies)
            print(f"{Fore.GREEN}Loaded {len(self.strategies)} Active Strategies.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading strategies: {e}{Style.RESET_ALL}")

    def _calculate_bins(self, df):
        # Placeholder for full binning logic (requires history)
        self.bins['volatility'] = [-np.inf, 50, 100, 150, 200, np.inf] # Mock bins
        pass 

    def _optimize_strategies(self, raw):
        import zlib
        optimized = []
        labels_vol = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
        
        self.magic_map = {} # Magic Int -> Strat UUID
        
        # Hardcode Legacy/Test mappings if needed
        # Mapping Test Order 123456 to 'inv_1p3' (d086...)
        test_strat_uuid = 'd0868224-a2a5-4ed5-8501-054e5882f319'
        self.magic_map[123456] = test_strat_uuid
        self.magic_map[999999] = test_strat_uuid # Cortex Test script
        
        for s in raw:
            filters = s.get('filters', {})
            strat_id = s.get('id')
            
            # Generate Stable Integer Magic ID (CRC32 of UUID)
            magic_id = zlib.crc32(strat_id.encode()) & 0xFFFFFFFF
            self.magic_map[magic_id] = strat_id
            
            # Parse Ratio
            lbl = s.get('ratio_label', 'TP 1x : SL 1x')
            tp_mult = 1.0
            sl_mult = 1.0
            try:
                parts = lbl.upper().replace('TP','').replace('SL','').replace(' ','').split(':')
                tp_mult = float(parts[0].replace('X',''))
                sl_mult = float(parts[1].replace('X',''))
            except: pass
            
            opt = {
                'id': strat_id, 
                'magic_id': magic_id, # Integer for MT5
                'type': s.get('strategy_type'),
                'name': s.get('name'),
                'hours': set(filters.get('hours', [])),
                'weekdays': set(filters.get('weekdays', [])),
                'direction': filters.get('direction', 'all'),
                'vol_indices': self._labels_to_indices(filters.get('volatility'), labels_vol),
                'tp_mult': tp_mult,
                'sl_mult': sl_mult
            }
            optimized.append(opt)
        return optimized

    def _labels_to_indices(self, selected_labels, all_labels):
        if not selected_labels: return set(range(len(all_labels)))
        return {i for i, label in enumerate(all_labels) if label in selected_labels}
    
    def get_bin_index(self, value, metric_name):
        if metric_name not in self.bins: return 0
        b = self.bins[metric_name]
        if len(b) < 2: return 0
        idx = np.digitize(value, b) - 1
        return max(0, min(idx, len(b) - 2))


class AssetNumpy:
    __slots__ = ['symbol', 'mgr', 'size', 'open', 'high', 'low', 'close', 
                 'sma20', 'volatility', 'range', 'atr', 'dist_sma', 'slope_deg', 
                 'distance_index', 'delta_di', 'daily_loss', 'is_locked', 'active_trades',
                 'active_strategy_ids']

    def __init__(self, symbol, initial_data, strategy_manager):
        self.symbol = symbol
        self.mgr = strategy_manager
        self.size = HISTORY_SIZE
        
        self.open = np.zeros(self.size)
        self.high = np.zeros(self.size)
        self.low = np.zeros(self.size)
        self.close = np.zeros(self.size)
        
        # Init Data if available
        if initial_data is not None:
             l = len(initial_data)
             if l >= self.size:
                 self.open[:] = initial_data['open'][-self.size:]
                 self.high[:] = initial_data['high'][-self.size:]
                 self.low[:] = initial_data['low'][-self.size:]
                 self.close[:] = initial_data['close'][-self.size:]
             else:
                 # Partial fill
                 self.close[:] = initial_data['close'].iloc[-1]
        else:
            self.close[:] = 100.0 # Mock Base

        self.sma20 = np.zeros(self.size)
        self.volatility = np.zeros(self.size)
        self.range = np.zeros(self.size)
        
        self.dist_sma = np.zeros(self.size)
        self.slope_deg = np.zeros(self.size)
        self.atr = np.zeros(self.size)
        self.distance_index = np.zeros(self.size)
        self.delta_di = np.zeros(self.size)
        
        self.daily_loss = 0.0
        self.is_locked = False
        self.active_trades = []
        self.active_strategy_ids = set() # Track IDs to enforce "One Trade Per Strategy"

        # Bootstrap Indicators
        self.update_indicators_full()

    def update_indicators_full(self):
        # Simple bootstrap for all indicators
        self.range = self.high - self.low
        self.sma20[:] = [np.mean(self.close[max(0, i-SMA_PERIOD):i+1]) for i in range(self.size)]
        self.volatility[:] = np.mean(self.range) 

    def update(self, open_, high, low, close, current_time, is_new_bar=False, current_price=None):
        if self.is_locked: return []
        
        # 1. Check TP/SL on Active Trades (Tick Level)
        current_p = current_price if current_price else close
        removed_trades = []
        pnl_change = 0.0
        
        for t in self.active_trades:
            # Check Exit
            hit = False
            pnl = 0.0
            
            if t.direction == 'Compra':
                if current_p >= t.tp_price:
                    pnl = (t.tp_price - t.entry_price)
                    hit = True
                    print(f"{Fore.GREEN}[WIN] {t.symbol} BUY TP! ID:{t.strategy_id} PnL:{pnl:.1f}{Style.RESET_ALL}")
                elif current_p <= t.sl_price:
                    pnl = (t.sl_price - t.entry_price) # Negative
                    hit = True
                    print(f"{Fore.RED}[LOSS] {t.symbol} BUY SL! ID:{t.strategy_id} PnL:{pnl:.1f}{Style.RESET_ALL}")
            elif t.direction == 'Venda':
                if current_p <= t.tp_price:
                    pnl = (t.entry_price - t.tp_price)
                    hit = True
                    print(f"{Fore.GREEN}[WIN] {t.symbol} SELL TP! ID:{t.strategy_id} PnL:{pnl:.1f}{Style.RESET_ALL}")
                elif current_p >= t.sl_price:
                    pnl = (t.entry_price - t.sl_price) # Negative
                    hit = True
                    print(f"{Fore.RED}[LOSS] {t.symbol} SELL SL! ID:{t.strategy_id} PnL:{pnl:.1f}{Style.RESET_ALL}")
            
            if hit:
                pnl_change += pnl
                removed_trades.append(t)
        
        if pnl_change != 0:
            self.update_pnl(pnl_change)
            
        for t in removed_trades:
            self.active_trades.remove(t)
            self.active_strategy_ids.discard(t.strategy_id) # Release Logic

        # 2. If it's NOT a new bar, we don't update signals (Minute Only request).
        if not is_new_bar:
            return []

        # 3. Handle NEW BAR (Roll & Calc)
        self.open[:-1] = self.open[1:]
        self.high[:-1] = self.high[1:]
        self.low[:-1] = self.low[1:]
        self.close[:-1] = self.close[1:]
        self.range[:-1] = self.range[1:]
        self.sma20[:-1] = self.sma20[1:]
        self.volatility[:-1] = self.volatility[1:]
        
        # Insert New Completed Candle
        self.open[-1] = open_
        self.high[-1] = high
        self.low[-1] = low
        self.close[-1] = close
        self.range[-1] = high - low
        
        # Recalc Indicators (Incremental)
        self.sma20[-1] = np.mean(self.close[-SMA_PERIOD:])
        self.volatility[-1] = np.mean(self.range[-SMA_PERIOD-1 : -1])
        self.dist_sma[-1] = abs(close - self.sma20[-1])
        
        # 4. Check Signals
        return self.check_signals_and_enter(current_time)

    def sync_positions(self):
        if self.mgr.mock_mode: return
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                print(f"{Fore.CYAN}Syncing {len(positions)} active positions from MT5...{Style.RESET_ALL}")
                for pos in positions:
                    # Map Position to ActiveTrade
                    direction = 'Compra' if pos.type == 0 else 'Venda'
                    magic = pos.magic
                    
                    # Resolve Strategy ID from Magic Map
                    strat_id = self.mgr.magic_map.get(magic)
                    
                    if strat_id:
                         # Find Name
                        strat_name = f"ID:{magic}"
                        for s in self.mgr.strategies:
                            if s['id'] == strat_id:
                                strat_name = s['name']
                                break
                    else:
                        strat_id = f"UNKNOWN_{magic}"
                        strat_name = "External"

                    t = ActiveTrade(self.symbol, direction, pos.price_open, pos.tp, pos.sl, 
                                    strat_id, strat_name, datetime.fromtimestamp(pos.time), ticket=pos.ticket)
                    
                    self.active_trades.append(t)
                    self.active_strategy_ids.add(strat_id)
                    print(f"{Fore.YELLOW}  -> Restored {direction} ({strat_name}) [Magic:{magic} -> ID:{strat_id[:8]}...] @ {pos.price_open}{Style.RESET_ALL}")
            else:
                print("No active positions found in MT5.")
        except Exception as e:
            print(f"{Fore.RED}Error syncing positions: {e}{Style.RESET_ALL}")


    def update_pnl(self, pnl):
        if pnl < 0:
            self.daily_loss += abs(pnl)
            if self.daily_loss >= MAX_LOSS_POINTS:
                self.is_locked = True
                print(f"{Fore.RED}[RISK] {self.symbol} DAILY LOSS LIMIT REACHED ({self.daily_loss}). LOCKED.{Style.RESET_ALL}")

    def check_signals_and_enter(self, current_time):
        c = self.close
        h = self.high
        l = self.low
        o = self.open
        sma = self.sma20
        # Volatility of the SIGNAL candle (Closed)
        vol = self.volatility[-2]
        
        # Current Metrics (Signal Candle is -2: Last Closed Candle)
        # Forensic used 'i-1'. i is current.
        slope = self.slope_deg[-2]
        
        executed = []
        
        # --- PRE-CALCULATE PATTERNS (Scalar) ---
        # 1. Inside Bar
        # Signal=-2, Mother=-3
        body_high_mother = max(o[-3], c[-3])
        body_low_mother = min(o[-3], c[-3])
        cond_inside = (o[-2] >= body_low_mother) and (o[-2] <= body_high_mother) and \
                      (c[-2] >= body_low_mother) and (c[-2] <= body_high_mother)

        # IB Buy
        # Fractal: c[-4]<c[-3] and c[-3]>c[-2] ? No.
        # Forensic: c3 < c2 > c1 (Indices relative to i).
        # i-1=Signal(-2), i-2=Mother(-3), i-3=PreMother(-4).
        # Fractal High at Mother(-3): (-4 < -3) & (-3 > -2)?
        # Forensic code: (df['c3'] < df['c2']) & (df['c2'] > df['c1'])
        # c1=shift(1) -> -2. c2=shift(2) -> -3. c3=shift(3) -> -4.
        cond_fractal_buy = (c[-4] < c[-3]) and (c[-3] > c[-2])
        cond_sma_buy = (c[-4] > sma[-4])
        ib_buy_sig = cond_fractal_buy and cond_sma_buy and cond_inside
        
        # IB Sell
        cond_fractal_sell = (c[-4] > c[-3]) and (c[-3] < c[-2])
        cond_sma_sell = (c[-4] < sma[-4])
        ib_sell_sig = cond_fractal_sell and cond_sma_sell and cond_inside

        # 2. Micro Channel
        # Indices: i-1(Signal, -2). 
        # Forensic: h4>h3>h2. Grab l1<l4. Trend c1>sma1.
        # l1 -> l[-2]. h4 -> h[-5].
        # wait. Forensic 'h4' is shift(4). Relative to i.
        # h1(shift 1) = -2.
        # h2(shift 2) = -3.
        # h3(shift 3) = -4.
        # h4(shift 4) = -5.
        
        # Buy
        cond_falling = (h[-5] > h[-4]) and (h[-4] > h[-3])
        cond_grab = (l[-2] < l[-5])
        cond_trend = (c[-2] > sma[-2])
        cond_slope_up = (sma[-2] > sma[-3])
        mc_buy_sig = cond_falling and cond_grab and cond_trend and cond_slope_up
        
        # Sell
        cond_rising = (h[-4] < h[-3]) and (h[-3] < h[-2])
        cond_trend_sell = (c[-2] < sma[-2])
        cond_slope_down = (sma[-2] < sma[-3])
        mc_sell_sig = cond_rising and cond_trend_sell and cond_slope_down
        
        
        for s in self.mgr.strategies:
            strat_id = s['id']
            # HEDGE RULE: One trade per Strategy ID.
            if strat_id in self.active_strategy_ids: continue
            
            if current_time.hour not in s['hours']: continue
            
            # --- FILTERS ---
            # Angle
            if 'angle_ranges' in s['filters']:
                ranges = s['filters']['angle_ranges']
                if not self.check_angle(slope, ranges): continue
                
            stype = s.get('strategy_type', 'unknown')
            
            # --- SIGNAL MATCH ---
            buy_pending = False
            sell_pending = False
            
            if stype == 'inside_bar':
                buy_pending = ib_buy_sig
                sell_pending = ib_sell_sig
            elif stype == 'micro_channel':
                buy_pending = mc_buy_sig
                sell_pending = mc_sell_sig
                
            # --- INVERSION & DIRECTION ---
            is_inverted = s['filters'].get('invert', False)
            final_action = None
            
            if buy_pending:
                final_action = 'Venda' if is_inverted else 'Compra'
            elif sell_pending:
                final_action = 'Compra' if is_inverted else 'Venda'
                
            # Check if this matches Strategy Direction
            want_direction = s['filters']['direction']
            
            # If we found a valid signal action, check if it's allowed
            accepted = False
            if final_action:
                if want_direction == 'all' or want_direction == final_action:
                    accepted = True
            
            if accepted:
                # --- RATIO & EXECUTION ---
                # Parse Ratio
                ratio_label = s.get('ratio_label', 'TP 1x : SL 1x')
                tp_mult, sl_mult = 1.0, 1.0
                try:
                    parts = ratio_label.split(':')
                    tp_part = parts[0].lower().replace('tp', '').replace('x', '').strip()
                    sl_part = parts[1].lower().replace('sl', '').replace('x', '').strip()
                    tp_mult, sl_mult = float(tp_part), float(sl_part)
                except: pass
                
                # Swap if Inverted
                if is_inverted:
                    tp_mult, sl_mult = sl_mult, tp_mult
                
                # Use c[-1] (Current Live Price) for execution reference if tick fails
                # Use vol[-2] for distance calc logic (Signal Volatility)
                self._execute_trade(s, c[-1], vol, current_time, 
                                    direction_override=final_action,
                                    tp_mult_override=tp_mult,
                                    sl_mult_override=sl_mult)
                                    
                executed.append(s['name'])
        
        return executed
    
    def check_angle(self, slope, ranges):
        if not ranges: return True
        abs_slope = abs(slope)
        for r in ranges:
            clean = r.replace('°', '')
            if '+' in clean:
                min_val = float(clean.replace('+', ''))
                if abs_slope >= min_val: return True
            elif '-' in clean:
                parts = clean.split('-')
                try:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    if min_val <= abs_slope < max_val: return True
                except: pass
        return False

    def _execute_trade(self, strategy, entry_price, vol, time_, direction_override=None, tp_mult_override=None, sl_mult_override=None):
        direction = direction_override if direction_override else strategy['direction']
        if direction == 'all': direction = 'Compra' # Default fallback if no override
        
        # Ensure correct defaults if not passed
        # But loop always passes them now.
        
        # Calculate TP/SL
        # Forensic used ref_range from previous candle (volatility[-1] is mean range?)  
        # realtime_processor init: self.volatility[:] = np.mean(self.range)... 
        # Yes, vol is mean range.
        
        strat_vol = vol # Base volatility
        
        tp_mult = tp_mult_override if tp_mult_override is not None else 1.0
        sl_mult = sl_mult_override if sl_mult_override is not None else 1.0
        
        dist_target = strat_vol * tp_mult
        dist_stop = strat_vol * sl_mult
        
        range_val = max(5.0, vol)
        tp_dist = range_val * strategy['tp_mult']
        sl_dist = range_val * strategy['sl_mult']
        
        # Determine Trade Type and Price
        if direction == 'Compra':
            order_type = mt5.ORDER_TYPE_BUY if not self.mgr.mock_mode else 0
            # For real execution, entry_price determined by Ask
            order_type = mt5.ORDER_TYPE_BUY
            price_exec = current_ask
            tp_price = price_exec + dist_target
            sl_price = price_exec - dist_stop
        else: # Venda
            order_type = mt5.ORDER_TYPE_SELL
            price_exec = current_bid
            tp_price = price_exec - dist_target
            sl_price = price_exec + dist_stop
            
        # Send Order
        try:
            # Map Strategy ID to Magic (Integer)
            strat_id_str = strategy['id']
            magic_id = self.mgr.magic_map.get(strat_id_str, 123456)
        
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": 1.0, # Fixed 1 lot
                "type": order_type,
                "price": price_exec,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": magic_id,
                "comment": f"{strategy['name'][:15]}",
                "type_time": mt5.ORDER_TIME_DAY,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            if not self.mgr.mock_mode:
                print(f"Sending Order: {direction} @ {price_exec} (TP:{tp_price} SL:{sl_price}) Magic:{magic_id}")
                result = mt5.order_send(req)
                
                # Robust Result Check
                if result is None:
                     print(f"{Fore.RED}[MT5 ERROR] Order Send returned None.{Style.RESET_ALL}")
                elif hasattr(result, 'retcode') and result.retcode != mt5.TRADE_RETCODE_DONE:
                     print(f"{Fore.RED}[MT5 ERROR] Order Failed: {result.comment} ({result.retcode}){Style.RESET_ALL}")
                elif isinstance(result, tuple) and result[0] != mt5.TRADE_RETCODE_DONE:
                     # Some versions return tuple
                     print(f"{Fore.RED}[MT5 ERROR] Order Failed (Tuple): {result}{Style.RESET_ALL}")
                else:
                    # Success
                    t_ticket = getattr(result, 'order', 0) if hasattr(result, 'order') else 0
                    print(f"{Fore.GREEN}[EXEC] {direction} executed! Ticket: {t_ticket}{Style.RESET_ALL}")
                    
                    # Track internally
                    t = ActiveTrade(self.symbol, direction, price_exec, tp_price, sl_price, 
                                    strat_id_str, strategy['name'], time_, ticket=t_ticket)
                    self.active_trades.append(t)
                    self.active_strategy_ids.add(strat_id_str)
                    
            else:
                 # Mock Success
                 print(f"{Fore.BLUE}[MOCK EXEC] {direction} {strategy['name']} @ {price_exec} TP:{tp_price} SL:{sl_price}{Style.RESET_ALL}")
                 t = ActiveTrade(self.symbol, direction, price_exec, tp_price, sl_price, 
                                 strategy['id'], strategy['name'], time_)
                 self.active_trades.append(t)
                 self.active_strategy_ids.add(strategy['id'])
                 
        except Exception as e:
            print(f"{Fore.RED}Exception in _execute_trade: {e}{Style.RESET_ALL}")


class RealtimeProcessorNumpy:
    def __init__(self):
        self.assets = []
        self.mgr = StrategyManager()
        self.symbol = "WING26" # Hardcoded
        self.mock_mode = not MT5_AVAILABLE
        
    def run(self):
        self.mgr.load_and_prepare()
        
        self.ensure_mt5_connection()

        print(f"Initializing Asset: {self.symbol}")
        init_data = None
        
        if not self.mock_mode:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, HISTORY_SIZE + 50)
            if rates is not None and len(rates) > 0:
                init_data = pd.DataFrame(rates)
                init_data['time'] = pd.to_datetime(init_data['time'], unit='s')
            else:
                print("Failed to fetch initial history. Retrying in Mock Mode temporarily.")
        
        self.assets = [AssetNumpy(self.symbol, init_data, self.mgr)]
        
        # Sync Existing Positions
        if not self.mock_mode:
            self.assets[0].sync_positions()

        
        print(f"{Fore.YELLOW}=== ENGINE STARTED (SINGLE ASSET | ID-BASED HEDGE) ==={Style.RESET_ALL}")
        
        last_minute = datetime.now().minute
        
        try:
            while True:
                now = datetime.now()
                is_new_minute = now.minute != last_minute
                
                curr_price = 0.0
                if not self.mock_mode:
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is None:
                        # Connection Lost Logic
                        print(f"{Fore.RED}Connection lost. Defaulting to last known price and attempting reconnect...{Style.RESET_ALL}")
                        self.ensure_mt5_connection() # Blocking reconnect attempt
                        tick = mt5.symbol_info_tick(self.symbol)
                        if tick: curr_price = tick.last
                        else: curr_price = self.assets[0].close[-1] # Fallback
                    else:
                        curr_price = tick.last
                else:
                    curr_price = self.assets[0].close[-1] + np.random.normal(0, 0.5)

                # 1. Update Ticks (TP/SL Check)
                self.assets[0].update(0,0,0,0, now, is_new_bar=False, current_price=curr_price)
                
                # 2. Update Candle (New Minute)
                if is_new_minute:
                    print(f"--- NEW MINUTE {now.strftime('%H:%M')} ---")
                    last_minute = now.minute
                    
                    if not self.mock_mode:
                        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 1, 1)
                        if rates is not None and len(rates) > 0:
                            r = rates[0]
                            self.assets[0].update(r['open'], r['high'], r['low'], r['close'], now, is_new_bar=True, current_price=curr_price)
                    else:
                        c = curr_price
                        self.assets[0].update(c, c+2, c-2, c, now, is_new_bar=True, current_price=c)
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            if MT5_AVAILABLE: mt5.shutdown()

    def ensure_mt5_connection(self):
        if not MT5_AVAILABLE:
            self.mock_mode = True
            self.mgr.mock_mode = True
            return

        if not mt5.initialize():
            print(f"{Fore.RED}MT5 Init Failed: {mt5.last_error()}. Retrying...{Style.RESET_ALL}")
            time.sleep(1)
            if not mt5.initialize():
                print(f"{Fore.RED}Retry Failed. Switching to Mock Mode.{Style.RESET_ALL}")
                self.mock_mode = True
                self.mgr.mock_mode = True
                return
        
        # Connected
        print(f"{Fore.GREEN}MT5 Connected.{Style.RESET_ALL}")
        if not mt5.symbol_select(self.symbol, True):
            print(f"Failed to select {self.symbol}.")
        
        self.mock_mode = False
        self.mgr.mock_mode = False


if __name__ == "__main__":
    RealtimeProcessorNumpy().run()
