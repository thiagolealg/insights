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

# Legacy/Test Mappings
magic_map[123456] = {'name': "Test Order / inv_1p3", 'type': "Manual/Test"}
magic_map[999999] = {'name': "Manual Test Script", 'type': "Manual/Test"}

try:
    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        raw_strategies = json.load(f)
        
    print(f"{Fore.CYAN}Loaded {len(raw_strategies)} strategies for verification.{Style.RESET_ALL}")
    
    for s in raw_strategies:
        strat_id = s.get('id')
        name = s.get('name')
        stype = s.get('strategy_type', 'Custom')
        if strat_id and name:
             # Generate Stable Integer Magic ID (CRC32 of UUID)
            magic_id = zlib.crc32(strat_id.encode()) & 0xFFFFFFFF
            magic_map[magic_id] = {'name': name, 'type': stype}
            
except Exception as e:
    print(f"{Fore.RED}Failed to load strategies: {e}{Style.RESET_ALL}")


if not mt5.initialize():
    print(f"MT5 Init Failed: {mt5.last_error()}")
    quit()

# Define Yesterday (or Today if desired, let's keep yesterday as requested but flexible)
now = datetime.now()
target_date = now - timedelta(days=1)
# target_date = now # Uncomment to check TODAY

from_date = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
to_date = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)

print(f"{Fore.CYAN}Fetching History for: {from_date.date()}{Style.RESET_ALL}")

# Get History
deals = mt5.history_deals_get(from_date, to_date)

with open('history_report.txt', 'w', encoding='utf-8') as f:
    if deals is None or len(deals) == 0:
        f.write(f"No deals found for {from_date.date()}.\n")
    else:
        f.write(f"Found {len(deals)} deals:\n")
        f.write(f"{'Time':<20} | {'Symbol':<8} | {'Type':<9} | {'Vol':<3} | {'Price':<9} | {'Profit':<8} | {'Magic':<15} | {'Strategy Name':<25} | {'Technique':<15}\n")
        f.write("-" * 130 + "\n")
        
        total_profit = 0.0
        stats = {}
        
        for deal in deals:
            # 0=Buy, 1=Sell
            type_str = "BUY" if deal.type == 0 else "SELL" if deal.type == 1 else str(deal.type)
            if deal.entry == mt5.DEAL_ENTRY_IN: type_str += " (IN)"
            elif deal.entry == mt5.DEAL_ENTRY_OUT: type_str += " (OUT)"
            
            dt_str = datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Resolve Info
            info = magic_map.get(deal.magic, {'name': f"UNKNOWN ({deal.magic})", 'type': '?'})
            strat_name = info['name']
            strat_type = info['type']
            
            if len(strat_name) > 25: strat_name = strat_name[:22] + "..."
            
            f.write(f"{dt_str:<20} | {deal.symbol:<8} | {type_str:<9} | {deal.volume:<3} | {deal.price:<9.2f} | {deal.profit:<8.2f} | {deal.magic:<15} | {strat_name:<25} | {strat_type:<15}\n")
            
            total_profit += deal.profit
            
            # Aggregation logic
            if deal.entry == mt5.DEAL_ENTRY_OUT: # Only count closed trades for stats
                if strat_name not in stats:
                    stats[strat_name] = {'type': strat_type, 'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
                
                stats[strat_name]['count'] += 1
                stats[strat_name]['pnl'] += deal.profit
                if deal.profit > 0:
                    stats[strat_name]['wins'] += 1
                else:
                    stats[strat_name]['losses'] += 1

        f.write("-" * 130 + "\n")
        f.write(f"Total Profit: {total_profit:.2f}\n\n")
        
        f.write("=== Strategy Performance (Closed Trades Only) ===\n")
        f.write(f"{'Strategy Name':<25} | {'Technique':<15} | {'Trades':<6} | {'Wins':<5} | {'Losses':<6} | {'Win Rate':<8} | {'PnL':<10}\n")
        f.write("-" * 100 + "\n")
        
        for name, s in stats.items():
            win_rate = (s['wins'] / s['count']) * 100 if s['count'] > 0 else 0.0
            f.write(f"{name:<25} | {s['type']:<15} | {s['count']:<6} | {s['wins']:<5} | {s['losses']:<6} | {win_rate:>6.1f}% | {s['pnl']:<10.2f}\n")



print("Report saved to history_report.txt")
mt5.shutdown()
