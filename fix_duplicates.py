
import json
import os

STRATEGIES_FILE = 'saved_strategies.json'

def fix_duplicates():
    if not os.path.exists(STRATEGIES_FILE):
        print("File not found!")
        return

    try:
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
            
        print(f"Loaded {len(strategies)} strategies.")
        
        # Group by name
        by_name = {}
        for s in strategies:
            name = s.get('name', 'Unknown')
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(s)
            
        to_remove_ids = []
        
        for name, duplicates in by_name.items():
            if len(duplicates) > 1:
                print(f"Checking duplicates for '{name}'...")
                
                # strategy to keep: prefer one with acc_levels
                has_acc = [s for s in duplicates if 'acc_levels' in s.get('filters', {}) and s['filters']['acc_levels']]
                no_acc = [s for s in duplicates if s not in has_acc]
                
                if has_acc and no_acc:
                    print(f"  Found valid strategy and duplicates without acc_levels.")
                    # Keep the first valid one, remove all invalid ones
                    # Or keep the most recent?
                    # Let's just remove the ones without acc_levels if we have one WITH acc_levels
                    for bad in no_acc:
                        print(f"  REMOVING duplicate ID: {bad['id']} (Missing acc_levels)")
                        to_remove_ids.append(bad['id'])
                elif len(duplicates) > 1:
                    print(f"  Duplicates found but cannot distinguish by acc_levels. Removing older duplicates, keeping newest.")
                    # Sort by created_at desc
                    duplicates.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                    # Keep first (newest), remove others
                    for bad in duplicates[1:]:
                        print(f"  REMOVING older duplicate ID: {bad['id']}")
                        to_remove_ids.append(bad['id'])

        if to_remove_ids:
            original_count = len(strategies)
            strategies = [s for s in strategies if s['id'] not in to_remove_ids]
            print(f"Removed {len(to_remove_ids)} strategies. Count: {original_count} -> {len(strategies)}")
            
            with open(STRATEGIES_FILE, 'w', encoding='utf-8') as f:
                json.dump(strategies, f, ensure_ascii=False, indent=4)
            print("File updated.")
        else:
            print("No duplicates removed.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_duplicates()
