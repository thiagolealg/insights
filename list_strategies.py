
import json
import os

STRATEGIES_FILE = 'saved_strategies.json'

def list_strategies():
    if not os.path.exists(STRATEGIES_FILE):
        print("File not found!")
        return

    try:
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
            
        print(f"Found {len(strategies)} strategies.")
        for s in strategies:
            name = s.get('name', 'Unknown')
            if 'inv' not in name.lower(): continue
            
            id_ = s.get('id', 'No ID')
            filters = s.get('filters', {})
            acc = filters.get('acc_levels', [])
            
            print(f"Name: {name}")
            print(f"ID: {id_}")
            print(f"Acc Levels ({len(acc)}): {acc}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_strategies()
