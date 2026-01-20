import json

try:
    with open('stability_report.json', 'r') as f:
        results = json.load(f)
    
    print("\nTOP 10 ESTRATEGIAS MAIS ESTAVEIS (2020-2026)")
    print("="*60)
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. {r['nome']}")
        print(f"   Tipo: {r['tipo']} | Total: {r['total']:,.0f}")
        print(f"   Anos Positivos: {r['anos_positivos']}/7 | Score: {r['score']:.1f}")
        lucros = [f"{l/1000:.0f}k" for l in r['lucros']]
        print(f"   Histórico: {lucros}")
        print("-" * 60)
        
except Exception as e:
    print(f"Erro: {e}")
