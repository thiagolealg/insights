
import json
import uuid
from datetime import datetime
from strategy_analyzer import StrategyAnalyzer

STRATEGIES_FILE = 'saved_strategies.json'
DATA_FILE = "win_full_data.parquet"

def restore_strategy():
    try:
        # 1. Load existing strategies causes
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            strategies = json.load(f)

        # 2. Define the lost strategy (Inside Bar version)
        # Using typical settings observed for "inv_1p4"
        new_strategy = {
            "id": str(uuid.uuid4()),
            "name": "inv_venda_1p4",
            "description": "Restored Auto-Tuned Strategy",
            "created_at": datetime.now().isoformat(),
            "strategy_type": "inside_bar",
            "ratio_label": "TP 1x : SL 4x",
            "filters": {
                "direction": "Venda",
                 # Importing common filters or leaving empty to be tuned?
                 # Better to copy basic filters from similar strategies if possible, 
                 # but for now we'll rely on the auto-tuner to fill in acc_levels.
                 # We'll set basic filters that define the "1p4" logic if known.
                "hours": [9, 10, 11, 12, 13, 14, 15, 16], # Broad assumption
                "acc_levels": [] # To be tuned
            }
        }
        
        # 3. Append and Save
        strategies.append(new_strategy)
        with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
            
        print(f"Restored strategy 'inv_venda_1p4' (inside_bar) with ID: {new_strategy['id']}")
        
        # 4. Immediate Auto-Tune for this specific ID
        print("Running Auto-Tune for recovered strategy...")
        
        analyzer = StrategyAnalyzer(DATA_FILE, strategy_type='inside_bar')
        analyzer.run_backtest()
        
        # Use TP 1x : SL 4x for tuning
        df = analyzer.get_result_for_filters("TP 1x : SL 4x", new_strategy['filters'])
        
        # Get stats by acc
        acc_stats = analyzer.get_stats_by_acc()
        
        selected_levels = []
        print("\nOptimization Results:")
        for row in acc_stats:
            sharpe = row.get('sharpe_ratio', 0)
            if sharpe > 1.7:
                selected_levels.append(row['acc_level'])
                print(f"  Selected: {row['acc_level']} (Sharpe: {sharpe})")
        
        if selected_levels:
            new_strategy['filters']['acc_levels'] = selected_levels
            
            # Save again with updated filters
            with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
                json.dump(strategies, f, ensure_ascii=False, indent=2)
            print("Strategy updated with optimized ACC levels.")
        else:
            print("No high-performance acc levels found > 1.7.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    restore_strategy()
