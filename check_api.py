import requests
try:
    r = requests.get('http://127.0.0.1:5000/api/strategies?strategy=inside_bar')
    print(f"Status: {r.status_code}")
    print(f"Content: {r.text[:500]}...")  # Print first 500 chars
except Exception as e:
    print(f"Error: {e}")
