import json
import os

STRATEGIES_FILE = 'saved_strategies.json'

if os.path.exists(STRATEGIES_FILE):
    try:
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"SUCCESS: Loaded {len(data)} strategies.")
    except Exception as e:
        print(f"ERROR: Failed to load JSON. Reason: {e}")
        # Tentar ler o arquivo para ver se tem lixo
        with open(STRATEGIES_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"First 100 chars: {content[:100]}")
else:
    print("File not found.")
