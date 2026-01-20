import json
import os

def update_years_in_strategies(filepath="saved_strategies.json"):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            strategies = json.load(f)
            
        updated_count = 0
        for s in strategies:
            if 'filters' in s and 'years' in s['filters']:
                years = s['filters']['years']
                # If they were filtering for 2024, assume they want the new years too
                if 2024 in years:
                    orig_len = len(years)
                    if 2025 not in years:
                        years.append(2025)
                    if 2026 not in years:
                        years.append(2026)
                    
                    if len(years) > orig_len:
                        s['filters']['years'] = sorted(list(set(years)))
                        updated_count += 1
                        
        if updated_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(strategies, f, ensure_ascii=False, indent=2)
            print(f"Updated {updated_count} strategies to include 2025 and 2026.")
        else:
            print("No strategies needed updating.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_years_in_strategies()
