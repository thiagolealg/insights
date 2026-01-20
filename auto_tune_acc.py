
import json
import os
import sys
import pandas as pd
from strategy_analyzer import StrategyAnalyzer

# Config
SHARPE_THRESHOLD = 1.7
STRATEGIES_FILE = 'saved_strategies.json'

def load_saved_strategies():
    if not os.path.exists(STRATEGIES_FILE):
        return []
    with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_strategies(strategies):
    with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(strategies, f, indent=4, ensure_ascii=False)

def auto_tune_strategies():
    print("🚀 Starting Auto-Tune for Delta T Acceleration (Sharpe > 1.7)...")
    
    # Ensure data is loaded
    analyzer = StrategyAnalyzer('win_full_data.parquet')
    analyzer.load_data()
    
    strategies = load_saved_strategies()
    if not strategies:
        print("⚠️ No saved strategies found.")
        return

    updated_count = 0
    
    for s in strategies:
        strategy_name = s.get('name', 'Unknown')
        strategy_type = s.get('strategy_type', 'inside_bar')
        print(f"\n🔍 Analyzing: {strategy_name} ({strategy_type})")
        
        # 1. Run Backtest for this strategy type
        analyzer.strategy_type = strategy_type
        # Apply ratio if present
        if 'ratio_label' in s:
            target_ratio = s['ratio_label']
            # Find in available
            ratio_config = next((r for r in StrategyAnalyzer.AVAILABLE_RATIOS if r['label'] == target_ratio), None)
            if ratio_config:
               analyzer.take_profit_ratio = ratio_config['tp']
               analyzer.stop_loss_ratio = ratio_config['sl']
        
        analyzer.run_backtest()
        
        # 2. Apply existing filters (EXCEPT existing acc_levels to see true performance of all levels)
        filters = s.get('filters', {}).copy()
        # Temporarily remove acc_levels to get stats for ALL levels
        if 'acc_levels' in filters:
            del filters['acc_levels']
            
        analyzer.apply_filters(filters)
        
        # 3. Get Stats by Acc Level
        stats = analyzer.get_stats_by_acc(use_filtered=True)
        
        if not stats:
            print("   ⚠️ No trades found for this configuration.")
            continue
            
        # 4. Filter Levels > Threshold
        best_levels = []
        print(f"   📊 Checking levels...")
        for row in stats:
            sharpe = row.get('sharpe_ratio', 0)
            level = row.get('acc_level')
            if sharpe > SHARPE_THRESHOLD:
                best_levels.append(level)
                print(f"      ✅ {level}: Sharpe {sharpe:.2f}")
            # else:
            #     print(f"      ❌ {level}: Sharpe {sharpe:.2f}")
        
        # 5. Update Strategy
        if best_levels:
            if 'filters' not in s:
                s['filters'] = {}
            
            # Check if meaningful change
            old_levels = s['filters'].get('acc_levels', [])
            # Sort for comparison
            best_levels.sort()
            old_levels.sort()
            
            if best_levels != old_levels:
                s['filters']['acc_levels'] = best_levels
                print(f"   💾 UPDATED: Set {len(best_levels)} levels.")
                updated_count += 1
            else:
                print("   👌 Already optimized.")
        else:
             print("   ⚠️ No levels met the threshold. Keeping existing (or all).")

    if updated_count > 0:
        save_strategies(strategies)
        print(f"\n✅ Auto-Tune Complete! Updated {updated_count} strategies.")
    else:
        print("\n✅ Auto-Tune Complete! No changes needed.")

if __name__ == "__main__":
    auto_tune_strategies()
