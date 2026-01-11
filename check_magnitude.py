from strategy_analyzer import StrategyAnalyzer
import json

def check_magnitude():
    analyzer = StrategyAnalyzer('attached_assets/win_1767085916180.txt')
    print("Starting fresh analysis...")
    analyzer.load_data()
    analyzer.run_backtest()
    print(f"Trades DF size: {len(analyzer.trades_df)}")
        
    stats = analyzer.get_stats_by_weekday()
    output = []
    output.append("Weekday Profits:")
    for s in stats:
        output.append(f"{s['weekday_name']} (ID {s['weekday']}): {s['total_profit']}")
        
    with open('magnitude_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    print("Done writing to magnitude_result.txt")

if __name__ == "__main__":
    check_magnitude()
