
import requests
import json

try:
    response = requests.get('http://localhost:5000/api/strategies')
    strategies = response.json()
    
    print(f"API returned {len(strategies)} strategies.")
    
    for s in strategies:
        if 'inv' in s['name'].lower() and 'venda' in s['name'].lower():
            print(f"Name: {s['name']}")
            print(f"ID: {s['id']}")
            # The list endpoint might not return full details (filters).
            # Let's fetch details.
            
            detail_resp = requests.get(f"http://localhost:5000/api/strategies/{s['id']}")
            try:
                details = detail_resp.json()
                filters = details.get('filters', {})
                acc = filters.get('acc_levels', [])
                print(f"Acc Levels ({len(acc)}): {acc}")
            except:
                print("Failed to get details.")
            print("-" * 30)
            
except Exception as e:
    print(f"Error: {e}")
