import requests
import json

BASE_URL = 'http://127.0.0.1:5000/api'

def test_flow():
    # 1. List strategies
    print("Listing strategies...")
    r = requests.get(f'{BASE_URL}/strategies')
    strategies = r.json()
    print(f"Found {len(strategies)} strategies.")
    
    strat_id = None
    if len(strategies) == 0:
        # Create dummy strategy
        print("Creating dummy strategy...")
        payload = {
            "name": "Test Strategy",
            "filters": {"direction": "Compra"}, # Filter by Buy only
            "ratio_label": "TP 1x : SL 2x"
        }
        r = requests.post(f'{BASE_URL}/strategies', json=payload)
        data = r.json()
        if data['status'] == 'ok':
            strat_id = data['strategy']['id']
            print(f"Created strategy {strat_id}")
    else:
        strat_id = strategies[0]['id']
        print(f"Using existing strategy {strat_id}")
        
    # 2. Apply Combined
    print(f"Applying combined strategy with ID {strat_id}...")
    payload = {"ids": [strat_id]}
    r = requests.post(f'{BASE_URL}/strategies/apply_combined', json=payload)
    print("Apply response:", r.text)
    
    if r.status_code == 200:
        res = r.json()
        print(f"Combined count: {res.get('count')}")
        
        # 3. Check summary filtered
        print("Checking filtered summary...")
        r = requests.get(f'{BASE_URL}/summary?filtered=true')
        print("Filtered Summary:", r.json())
    else:
        print("Failed to apply combined.")

if __name__ == '__main__':
    try:
        test_flow()
    except Exception as e:
        print(e)
