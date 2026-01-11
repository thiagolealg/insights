from strategy_analyzer import StrategyAnalyzer
import os

DATA_FILE = "attached_assets/win_1767085916180.txt"

print("Checking file...", os.path.exists(DATA_FILE))
try:
    s = StrategyAnalyzer(DATA_FILE, strategy_type='inside_bar')
    print("Running backtest...")
    s.run_backtest()
    print("Done. Trades:", len(s.trades))
    print("Errors:", s.all_trades_df.empty)
except Exception as e:
    print("CRITICAL ERROR:", e)
    import traceback
    traceback.print_exc()
