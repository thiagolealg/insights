# Trading Insights - Plataforma de Backtesting

Sistema de backtesting e análise de estratégias de trading para o mercado brasileiro (WIN - Mini Índice).

## 📊 Visão Geral

Plataforma web para backtesting de estratégias de Price Action com filtros avançados e análise estatística detalhada.

## 🚀 Tecnologias

- **Backend:** Python 3.10+, Flask
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Dados:** Pandas, NumPy, Parquet (otimizado)
- **Cache:** Pickle para backtests pré-calculados

## 📁 Estrutura do Projeto

```
Trading-Insights/
├── app.py                    # Servidor Flask (API + rotas)
├── strategy_analyzer.py      # Engine de backtesting
├── saved_strategies.json     # Estratégias salvas (portfólio)
├── templates/
│   └── index.html           # Dashboard principal
├── attached_assets/
│   └── win_*.txt            # Dados históricos (CSV/Parquet)
└── .cache/                  # Cache de backtests
```

## 🎯 Estratégias Disponíveis

### 1. Inside Bar
Detecta candles "engolidos" pelo anterior:
- **Compra:** `High[i-1] < High[i-2]` e `Low[i-1] > Low[i-2]` + Close > SMA(20)
- **Venda:** Mesma lógica + Close < SMA(20)

### 2. Micro Channel
Detecta sequência de reversão (6 candles contra + 3 a favor):
- **Compra:** 6 bears + 3 bulls + Close abaixo da SMA
- **Venda:** 6 bulls + 3 bears + Close acima da SMA

### 3. Reversão Desvio Padrão (STD Reversal)
Detecta exaustão via volatilidade:
- **Compra (Fundo):** `STD(H,L,C) > (High - Close) * 1.2` por 4 candles
- **Venda (Topo):** `STD(H,L,C) > (Close - Low) * 1.2` por 4 candles

### 4. SMA Trend
Estratégia baseada em tendência da média móvel:
- **Compra:** Sequência específica de candles + SMA ascendente
- **Venda:** Sequência inversa + SMA descendente

### 5. Bull Bear
Padrão clássico de reversão:
- 6 candles de tendência + 3 candles de reversão

## 🔧 Filtros Disponíveis

| Filtro | Descrição |
|--------|-----------|
| **Direção** | Compra, Venda ou Todas |
| **Horários** | 9h às 18h |
| **Dias da Semana** | Segunda a Sexta |
| **Anos** | 2008-2024 |
| **Volatilidade** | Muito Baixa → Muito Alta (5 níveis) |
| **Ângulo SMA** | 0-5° até 90°+ (19 faixas) |
| **Distância SMA** | D1-D10 (decis) |
| **Índice de Distância** | DI1-DI10 (preço dia anterior) |
| **Take/Stop** | 7 proporções (1:4 até 4:1) |
| **Inverter** | Inverte direção da operação |

## 📈 Métricas Calculadas

- **Win Rate:** Taxa de acerto
- **Lucro Total:** Soma dos resultados
- **Sharpe Ratio:** Retorno ajustado ao risco
- **Profit Factor:** Ganhos / Perdas
- **Max Drawdown:** Maior queda do capital
- **Avg RR:** Risk/Reward médio

## 💾 Sistema de Portfólio

### Salvar Estratégia
1. Configure os filtros desejados
2. Digite um nome no campo
3. Clique "Salvar Filtros Atuais"

### Combinar Estratégias
1. Clique nos cartões para selecionar (borda verde)
2. Clique "Calcular Performance Geral"
3. Visualize estatísticas combinadas

### Ações nos Cartões
- 👁️ **Carregar:** Aplica os filtros salvos
- 📊 **Analisar:** Calcula estatísticas individuais
- 📋 **Copiar:** Copia JSON para clipboard
- 🗑️ **Excluir:** Remove a estratégia

## 🖥️ Como Executar

```bash
# Instalar dependências
pip install flask pandas numpy pyarrow

# Executar
python app.py

# Acessar
http://localhost:5000
```

## ⚡ Otimizações de Performance

1. **Parquet:** Leitura ~10x mais rápida que CSV
2. **Cache Pickle:** Evita recálculo de backtests
3. **Vetorização:** Cálculos com NumPy/Pandas
4. **Multi-threading:** Inicialização paralela de estratégias

## 📊 Dados Históricos

- **Instrumento:** WIN$ (Mini Índice Bovespa)
- **Timeframe:** 1 minuto
- **Período:** 2008-2024
- **Total:** ~2 milhões de candles
- **Colunas:** time, open, high, low, close, tick_volume, spread, real_volume

## 🔄 Versionamento de Cache

O sistema usa versionamento de cache (`_v20`) para garantir recálculo quando a lógica muda. Incrementar a versão força regeneração dos backtests.

---

**Desenvolvido para análise quantitativa de estratégias de trading.**
