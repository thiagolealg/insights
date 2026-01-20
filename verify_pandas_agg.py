import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import json
import zlib
from colorama import init, Fore, Style

init()

# 1. Load Strategy Map
STRATEGIES_FILE = "saved_strategies.json"
magic_map = {}
magic_map[123456] = "Test Order / inv_1p3"
magic_map[999999] = "Manual Test Script"

try:
    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        raw_strategies = json.load(f)
    for s in raw_strategies:
        strat_id = s.get('id')
        name = s.get('name')
        if strat_id and name:
            magic_id = zlib.crc32(strat_id.encode()) & 0xFFFFFFFF
            magic_map[magic_id] = name
except: pass

if not mt5.initialize():
    quit()

# Yesterday
now = datetime.now()
target_date = now - timedelta(days=1)
# target_date = now # Uncomment for TODAY

from_date = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
to_date = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)

deals = mt5.history_deals_get(from_date, to_date)
mt5.shutdown()

if not deals:
    print("No deals.")
    quit()

# Convert to DataFrame
df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
df['time'] = pd.to_datetime(df['time'], unit='s')

# Filter for OUT deals (Closed Trades) for PnL
closed_trades = df[df['entry'] == mt5.DEAL_ENTRY_OUT].copy()

if closed_trades.empty:
    print("No closed trades found.")
    quit()

# Map Names
closed_trades['strategy'] = closed_trades['magic'].apply(lambda x: magic_map.get(x, f"UNKNOWN_{x}"))

# Group By
summary = closed_trades.groupby('strategy')['profit'].agg(['count', 'sum'])
summary.columns = ['Trades', 'PnL']

# Calculate Wins/Losses
wins = closed_trades[closed_trades['profit'] > 0].groupby('strategy')['profit'].count()
losses = closed_trades[closed_trades['profit'] <= 0].groupby('strategy')['profit'].count()

summary['Wins'] = wins
summary['Losses'] = losses
summary.fillna(0, inplace=True)
summary['Win Rate'] = (summary['Wins'] / summary['Trades']) * 100

print(f"\n{Fore.CYAN}=== PANDAS INDEPENDENT CALCULATION ({target_date.date()}) ==={Style.RESET_ALL}")
print(summary.sort_values(by='PnL', ascending=False))
print(f"\n{Fore.GREEN}Total Verified PnL: {summary['PnL'].sum():.2f}{Style.RESET_ALL}")
