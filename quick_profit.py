import requests

# Verificar status
status = requests.get('http://localhost:5000/api/status').json()
print(f"Status bull_bear: {status['strategies']['bull_bear']}")

# Aplicar filtro 2026
r = requests.post('http://localhost:5000/api/apply_filters', 
    json={'strategy': 'bull_bear', 'years': [2026]})
print(f"Filtro: {r.json()}")

# Buscar curva de equity (mostra lucro acumulado)
r2 = requests.get('http://localhost:5000/api/equity_curve?strategy=bull_bear&filtered=true')
equity = r2.json()
if equity:
    ultimo = equity[-1] if equity else {}
    print(f"\n=== RESULTADO 2026 ===")
    print(f"Lucro acumulado: {ultimo.get('cumulative_result', 'N/A')} pontos")
    print(f"Ultimo trade: {ultimo.get('entry_time', 'N/A')}")
else:
    print("Sem dados de equity")

# Buscar summary
r3 = requests.get('http://localhost:5000/api/summary?strategy=bull_bear&filtered=true')
s = r3.json()
print(f"\nTrades: {s.get('total_trades')}")
print(f"Win Rate: {s.get('win_rate')}%")
print(f"Total Profit: {s.get('total_profit')}")
print(f"Avg Profit: {s.get('avg_profit')}")
