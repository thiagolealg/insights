import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import zlib
from colorama import init, Fore, Style

init()

# 1. Load Strategies
STRATEGIES_FILE = "saved_strategies.json"
strategies = []
try:
    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        strategies = json.load(f)
    print(f"{Fore.CYAN}Loaded {len(strategies)} strategies.{Style.RESET_ALL}")
except Exception as e:
    print(f"Error loading strategies: {e}")
    quit()

# 2. Fetch Data
if not mt5.initialize(): quit()

now = datetime.now()
target_date = now - timedelta(days=1)
# target_date = now # Uncomment for TODAY

from_date = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
to_date = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)
symbol = "WING26"

print(f"Fetching Data for {symbol} on {target_date.date()}...")
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_date, to_date)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("No data found.")
    quit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df['close'] = df['close'].astype(float)
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)

# 3. Indicators
# -------------------------------------------------------------------------
df['sma20'] = df['close'].rolling(window=20).mean()
df['range'] = df['high'] - df['low']

# ATR (Simple Rolling Mean of Range, window 20)
# Note: Should technically be shift(1) relative to decision time? 
# In vectorized logic, we usually use metrics available at 'i-1'.
# Calculating signals row-by-row based on 'open time' i.
# So we need metrics from the CLOSED candle i-1.
df['atr'] = df['range'].rolling(window=20).mean().fillna(1.0) # shift(1) done below via usage df.iloc[i-1] logic variants

# Shifts for Signal Logic (i-1 is the 'Signal' candle, just closed)
df['c1'] = df['close'].shift(1)
df['c2'] = df['close'].shift(2)
df['c3'] = df['close'].shift(3) 

df['o1'] = df['open'].shift(1)
df['o2'] = df['open'].shift(2)

df['sma3'] = df['sma20'].shift(3)

# Filter Metrics (Shift 1 = Last Closed Candle)
df['sma20_shift1'] = df['sma20'].shift(1)
df['sma20_shift6'] = df['sma20'].shift(6) # for Slope (i-1) - (i-6) = 5 bars
df['atr_shift1'] = df['atr'].shift(1)

# Angle / Slope Degrees Logic
dy = df['sma20_shift1'] - df['sma20_shift6']
norm_slope = ((dy / 5.0) / df['atr_shift1']) * 10.0
df['slope_deg'] = np.degrees(np.arctan(norm_slope))

# Shifts for Micro Channel
df['h1'] = df['high'].shift(1)
df['h2'] = df['high'].shift(2)
df['h3'] = df['high'].shift(3)
df['h4'] = df['high'].shift(4)
df['l1'] = df['low'].shift(1)
df['l4'] = df['low'].shift(4)
df['sma1'] = df['sma20'].shift(1)
df['sma2'] = df['sma20'].shift(2)


# --- STRATEGY 1: INSIDE BAR ---
# Body calc
df['body_high_2'] = np.maximum(df['o2'], df['c2'])
df['body_low_2'] = np.minimum(df['o2'], df['c2'])

cond_inside = (df['o1'] >= df['body_low_2']) & (df['o1'] <= df['body_high_2']) & \
              (df['c1'] >= df['body_low_2']) & (df['c1'] <= df['body_high_2'])

# Buy
cond_fractal_buy = (df['c3'] < df['c2']) & (df['c2'] > df['c1'])
cond_sma_buy = (df['c3'] > df['sma3'])
df['ib_buy'] = cond_fractal_buy & cond_sma_buy & cond_inside

# Sell
cond_fractal_sell = (df['c3'] > df['c2']) & (df['c2'] < df['c1'])
cond_sma_sell = (df['c3'] < df['sma3'])
df['ib_sell'] = cond_fractal_sell & cond_sma_sell & cond_inside


# --- STRATEGY 2: MICRO CHANNEL ---
# Buy: Falling Highs, Liquidity Grab, Trend Up, Slope Up
cond_falling = (df['h4'] > df['h3']) & (df['h3'] > df['h2'])
cond_grab = (df['l1'] < df['l4'])
cond_trend = (df['c1'] > df['sma1'])
cond_slope = (df['sma1'] > df['sma2'])
# Note: Original logic also checked slope > 0 explicitly? Assuming sma1 > sma2 handles it.
df['mc_buy'] = cond_falling & cond_grab & cond_trend & cond_slope

# Sell: Rising Highs, Trend Down, Slope Down
cond_rising = (df['h3'] < df['h2']) & (df['h2'] < df['h1'])
cond_trend_sell = (df['c1'] < df['sma1'])
cond_slope_sell = (df['sma1'] < df['sma2'])
df['mc_sell'] = cond_rising & cond_trend_sell & cond_slope_sell

# Helper for Filters
def check_angle(slope, ranges):
    if not ranges: return True
    abs_slope = abs(slope)
    for r in ranges:
        # Parse '60-65°' or '90°+'
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

# Helper: Parse Ratio
def parse_ratio(label):
    # Example: "TP 1x : SL 3x"
    try:
        parts = label.split(':')
        tp_part = parts[0].lower().replace('tp', '').replace('x', '').strip()
        sl_part = parts[1].lower().replace('sl', '').replace('x', '').strip()
        return float(tp_part), float(sl_part)
    except:
        return 1.0, 1.0

# Helper: Simulate Trade
def simulate_trade(start_idx, direction, entry_price, tp_price, sl_price, df_slice):
    # df_slice should be df.iloc[start_idx:]
    for _, candle in df_slice.iterrows():
        h, l = candle['high'], candle['low']
        
        if direction == 'Compra':
            if h >= tp_price: return 'WIN', tp_price
            if l <= sl_price: return 'LOSS', sl_price
        else: # Venda
            if l <= tp_price: return 'WIN', tp_price
            if h >= sl_price: return 'LOSS', sl_price
            
    return 'OPEN', entry_price # End of day

# 4. Check Against Strategies
# Prepare Stats
stats = {}

with open('forensic_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"=== FORENSIC SIGNAL & PnL CHECK ({target_date.date()}) ===\n")
    f.write(f"{'Time':<20} | {'Strategy':<25} | {'Type':<12} | {'Angle':<6} | {'Sig':<4} | {'Result':<5} | {'PnL (R$)'}\n")
    f.write("-" * 110 + "\n")

    matches = 0

    for i, row in df.iterrows():
        if i < 20: continue # Warmup
        
        current_time = row['time']
        hour = current_time.hour
        slope = row['slope_deg']
        if pd.isna(slope): slope = 0.0
        
        # Check each strategy
        for s in strategies:
            stype = s.get('strategy_type', 'unknown')
            name = s['name']
            
            # Filter: Hours
            if hour not in s['filters']['hours']: continue
            
            # Filter: Direction
            direction_filter = s['filters']['direction']
            
            # Filter: Angle
            if 'angle_ranges' in s['filters']:
                if not check_angle(slope, s['filters']['angle_ranges']): continue

            # Signal Logic
            has_signal = False
            sig_type = ""
            
            if stype == 'inside_bar':
                buy_sig = row['ib_buy']
                sell_sig = row['ib_sell']
            elif stype == 'micro_channel':
                buy_sig = row['mc_buy']
                sell_sig = row['mc_sell']
            else:
                buy_sig, sell_sig = False, False
            
            # --- INVERSION LOGIC ---
            is_inverted = s['filters'].get('invert', False)
            
            final_action = None
            
            if buy_sig:
                final_action = 'Venda' if is_inverted else 'Compra'
            elif sell_sig:
                final_action = 'Compra' if is_inverted else 'Venda'
            
            # Filter: Direction (Matches the DESIRED trade direction)
            # If strategy says direction="Venda", we want final_action="Venda"
            want_direction = s['filters']['direction']
            
            if final_action:
                if want_direction == 'all' or want_direction == final_action:
                     has_signal = True
                     sig_type = final_action
            
            if has_signal:
                matches += 1
                
                # --- SIMULATION ---
                entry_price = row['open'] # Execution at Open of 'i'
                
                # Reference Range for Target (use i-1)
                ref_range = df['range'].iloc[i-1] 
                if ref_range <= 0: ref_range = 5.0
                
                # Ratio
                tp_mult, sl_mult = parse_ratio(s.get('ratio_label', 'TP 1x : SL 1x'))
                
                # If Inverted, Swap TP/SL logic (To win what was lost)
                if is_inverted:
                    tp_mult, sl_mult = sl_mult, tp_mult
                
                dist_target = ref_range * tp_mult
                dist_stop = ref_range * sl_mult
                
                # Stops Compliance (Min 30pts)
                # dist_target = max(30.0, dist_target) # Strict verification doesn't usually apply this unless coded
                # But 'realtime_processor' enforces it. Let's assume raw strategy first.
                
                if sig_type == 'Compra':
                    tp_price = entry_price + dist_target
                    sl_price = entry_price - dist_stop
                else:
                    tp_price = entry_price - dist_target
                    sl_price = entry_price + dist_stop
                    
                # Run Sim
                outcome, exit_price = simulate_trade(i, sig_type, entry_price, tp_price, sl_price, df.iloc[i:])
                
                # Calc PnL
                pts = 0.0
                if sig_type == 'Compra': pts = exit_price - entry_price
                else: pts = entry_price - exit_price
                
                money = pts * 0.20
                
                # Log
                f.write(f"{current_time} | {name:<25} | {stype:<12} | {slope:<6.1f} | {sig_type[0]:<4} | {outcome:<5} | {money:>8.2f}\n")
                
                # Stats
                if name not in stats: stats[name] = {'count':0, 'wins':0, 'pnl':0.0}
                stats[name]['count'] += 1
                stats[name]['pnl'] += money
                if outcome == 'WIN': stats[name]['wins'] += 1

    f.write("-" * 110 + "\n")
    f.write(f"Total Theoretical Signals: {matches}\n\n")
    
    # Table Results
    f.write("=== THEORETICAL PERFORMANCE TABLE ===\n")
    f.write(f"{'Strategy':<30} | {'Trades':<6} | {'Wins':<5} | {'Win Rate':<8} | {'PnL (R$)'}\n")
    f.write("-" * 80 + "\n")
    
    total_pnl = 0.0
    for name, dat in stats.items():
        wr = (dat['wins'] / dat['count']) * 100
        f.write(f"{name:<30} | {dat['count']:<6} | {dat['wins']:<5} | {wr:>6.1f}% | {dat['pnl']:>8.2f}\n")
        total_pnl += dat['pnl']
        
    f.write("-" * 80 + "\n")
    f.write(f"TOTAL THEORETICAL PnL: R$ {total_pnl:.2f}\n")

print(f"Forensic report saved to forensic_report.txt. Total PnL: {total_pnl:.2f}")
