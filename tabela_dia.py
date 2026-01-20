import requests
from datetime import datetime

# Buscar trades do dia filtrado para 2026 (que basicamente é só hoje)
r = requests.post('http://localhost:5000/api/apply_filters', 
    json={'strategy': 'bull_bear', 'years': [2026]})

r2 = requests.get('http://localhost:5000/api/trades?strategy=bull_bear&filtered=true')
trades = r2.json()

print(f"\n{'='*70}")
print(f"  TABELA DE TRADES - 12/01/2026 (Alta-Baixa)")
print(f"{'='*70}")
print(f"{'Hora':<12} {'Direção':<10} {'Entrada':<12} {'Saída':<12} {'Resultado':<12} {'Status'}")
print(f"{'-'*70}")

total = 0
winners = 0
for t in trades:
    hora = t.get('entry_time', '')[-8:-3] if t.get('entry_time') else 'N/A'
    direcao = t.get('direction', 'N/A')
    entrada = float(t.get('entry_price', 0))
    saida = float(t.get('exit_price', 0))
    resultado = float(t.get('result', 0))
    winner = t.get('winner', 'False') == 'True'
    status = '✅ WIN' if winner else '❌ LOSS'
    
    total += resultado
    if winner:
        winners += 1
    
    print(f"{hora:<12} {direcao:<10} {entrada:<12,.0f} {saida:<12,.0f} {resultado:>+10,.0f}  {status}")

print(f"{'-'*70}")
print(f"TOTAL: {len(trades)} trades | Winners: {winners} | Win Rate: {winners/len(trades)*100:.1f}%")
print(f"LUCRO DO DIA: {total:>+,.0f} pontos")
print(f"{'='*70}")
