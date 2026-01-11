from strategy_analyzer import StrategyAnalyzer
import json

def check_api():
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    analyzer.load_cache() # Should load v4
    
    # 1. Check Weekday Stats
    print("Checking Weekday Stats...")
    stats = analyzer.get_stats_by_weekday()
    wed_stat = next((s for s in stats if s['weekday'] == 2), None)
    
    if wed_stat:
        print(f"Wednesday Found: {wed_stat}")
        print(f"Profit: {wed_stat['total_profit']}")
    else:
        print("Wednesday NOT FOUND in stats!")
        
    # 2. Check Recent Trades
    print("\nChecking Recent Trades...")
    trades = analyzer.get_recent_trades(n=5)
    
    print("Columns in first trade:")
    if trades:
        print(trades[0].keys())
        print("Sample Trade Values:")
        print(trades[0])
        
        # Check for non-zero slope
        slopes = [float(t['slope_degrees']) for t in trades if t.get('slope_degrees')]
        print(f"Slopes found: {slopes}")
    else:
        print("No trades returned")

if __name__ == "__main__":
    check_api()
