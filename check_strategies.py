import requests
import json

BASE_URL = 'http://localhost:5000/api/strategies'

def test_strategies():
    # 1. List existing
    print("Listing strategies...")
    r = requests.get(BASE_URL)
    if r.status_code != 200:
        print(f"Failed to list: {r.text}")
        return
    existing = r.json()
    print(f"Found {len(existing)} strategies.")

    # 2. Setup Test Data
    new_strategy = {
        'name': 'Test Strategy Auto',
        'description': 'Created by verification script',
        'filters': {'direction': 'Compra'},
        'ratio_label': 'TP 1x : SL 1x',
        'strategy_type': 'inside_bar'
    }

    # 3. Create (Save)
    print("\nAttempting to SAVE strategy...")
    r = requests.post(BASE_URL, json=new_strategy)
    if r.status_code == 200:
        data = r.json()
        sid = data['strategy']['id']
        print(f"Save SUCCESS. ID: {sid}")
        
        # 4. Verify it's in the list
        r = requests.get(BASE_URL)
        current = r.json()
        if any(s['id'] == sid for s in current):
             print("Verified: Strategy is in the list.")
        else:
             print("ERROR: Saved strategy NOT found in list!")
             
        # 5. Update
        print(f"\nAttempting to UPDATE strategy {sid}...")
        update_data = {
            'name': 'Updated Name Verification',
            'filters': {'direction': 'Venda', 'hours': [9, 10]}
        }
        r = requests.put(f"{BASE_URL}/{sid}", json=update_data)
        if r.status_code == 200:
             print("Update SUCCESS.")
             print("Update SUCCESS.")
             # Verify update via DETAIL endpoint (List endpoint doesn't return filters)
             r = requests.get(f"{BASE_URL}/{sid}")
             updated_strat = r.json()
             
             name_ok = updated_strat and updated_strat['name'] == 'Updated Name Verification'
             filters_ok = updated_strat and updated_strat.get('filters', {}).get('direction') == 'Venda'
             
             if name_ok and filters_ok:
                 print("Verified: Name AND Filters updated correctly.")
             else:
                 print(f"ERROR: Update mismatch! Name={updated_strat['name']}, Dir={updated_strat['filters'].get('direction')}")
        else:
             print(f"Update FAILED: {r.text}")

        # 6. Delete
        print(f"\nAttempting to DELETE strategy {sid}...")
        r = requests.delete(f"{BASE_URL}/{sid}")
        if r.status_code == 200:
            print("Delete SUCCESS.")
            
            # 6. Verify deletion
            r = requests.get(BASE_URL)
            current = r.json()
            if not any(s['id'] == sid for s in current):
                 print("Verified: Strategy removed from list.")
            else:
                 print("ERROR: Strategy STILL in list after delete!")
        else:
            print(f"Delete FAILED: {r.text}")
            
    else:
        print(f"Save FAILED: {r.text}")

if __name__ == "__main__":
    test_strategies()
