
import json
import os

STRATEGIES_FILE = 'saved_strategies.json'

def find_strategy():
    if not os.path.exists(STRATEGIES_FILE):
        print("File not found!")
        return

    try:
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
            
        print(f"File contains {len(strategies)} strategies.")
        
        target_name = "inv_venda_1p4"
        found = False
        
        for i, s in enumerate(strategies):
            # Check name case-insensitive
            s_name = s.get('name', 'Unknown')
            if target_name in s_name.lower():
                print(f"[{i}] {s_name} ({s.get('id')})")
                filters = s.get('filters', {})
                if 'acc_levels' in filters:
                     print(f"   Acc Levels: {len(filters['acc_levels'])} items")
                else:
                     print("   Acc Levels: MISSING")
                print("-" * 10)
                found = True
                
        if not found:
            print(f"Strategy '{target_name}' NOT FOUND in file.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_strategy()
