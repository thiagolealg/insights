# Estratégia Inside Bar com Micro Trend - Análise de Trading

## Overview
Site de análise estatística para estratégia de trading baseada em Inside Bar com Micro Trend. Processa dados históricos do mini índice WIN e apresenta estatísticas detalhadas de performance.

## Estratégia Implementada
A estratégia inclui operações de **Compra** e **Venda** com lógica invertida:

### Compra (Micro Trend Up)
- 3 fechamentos consecutivos em alta: close[i-3] < close[i-2] < close[i-1]
- Inside Body: corpo do candle i-1 contido em i-2
- Filtro de tendência: close[i-3] > SMA(20)

### Venda (Micro Trend Down)
- 3 fechamentos consecutivos em baixa: close[i-3] > close[i-2] > close[i-1]
- Inside Body: corpo do candle i-1 contido em i-2
- Filtro de tendência: close[i-3] < SMA(20)

### Parâmetros (ambas direções)
- Entrada: close do candle i-1
- Stop Loss: entry_price +/- range(candle i-1)
- Take Profit: entry_price -/+ range(candle i-1)
- Saída: TP, SL ou close do candle i (1 barra após entrada)

## Filtros Interativos
O site permite filtrar os resultados por:
- Direção (Compra, Venda ou ambas)
- Inverter Operação (Stop vira Take, Take vira Stop)
- Horários do dia (9h-18h)
- Dias da semana (Seg-Sex)
- Anos (2008-2024)
- Volatilidade (5 níveis: Muito Baixa, Baixa, Média, Alta, Muito Alta)

Todos os filtros atualizam automaticamente:
- Resumo geral
- Estatísticas por direção
- Gráficos de horário, dia, volatilidade e curva de capital
- Tabelas de performance por ano e mês

## Estatísticas Disponíveis
- Resumo geral (total trades, win rate, lucro, Sharpe ratio, profit factor, drawdown)
- Performance por horário do dia
- Performance por dia da semana
- Performance por nível de volatilidade
- Performance por ano
- Performance por mês
- Curva de capital acumulado

## Arquitetura do Projeto

### Backend (Python/Flask)
- `app.py` - Servidor Flask com API REST
- `strategy_analyzer.py` - Lógica da estratégia e cálculo de estatísticas

### Frontend
- `templates/index.html` - Interface com gráficos Chart.js
- `static/style.css` - Estilos do dashboard

### Dados
- `attached_assets/win_1767085916180.txt` - Dados históricos do WIN (2+ milhões de barras)

## API Endpoints
- `GET /` - Página principal
- `GET /api/status` - Status de carregamento
- `GET /api/summary` - Resumo geral (suporta ?filtered=true)
- `GET /api/by_direction` - Estatísticas por direção (Compra/Venda)
- `GET /api/by_hour` - Estatísticas por horário
- `GET /api/by_weekday` - Estatísticas por dia da semana
- `GET /api/by_year` - Estatísticas por ano
- `GET /api/by_month` - Estatísticas por mês
- `GET /api/by_volatility` - Estatísticas por volatilidade
- `GET /api/equity_curve` - Curva de capital
- `GET /api/available_filters` - Horários e dias disponíveis
- `POST /api/apply_filters` - Aplica filtros (direction, hours, weekdays)
- `POST /api/reset_filters` - Remove todos os filtros

## Resultados da Estratégia com SMA(20) (2008-2024)
- **Total**: 1,075 trades (549 compras + 526 vendas)
- **Compra**: -570 pts, 44.44% win rate, Sharpe -0.47
- **Venda**: +430 pts, 41.25% win rate, Sharpe 0.36
- O filtro SMA(20) reduziu significativamente o número de trades

## Como Testar Outras Estratégias
Para testar outras estratégias, modifique o método `check_entry_conditions()` em `strategy_analyzer.py`. O sistema foi projetado para ser reproduzível - basta alterar as condições de entrada.

## Tecnologias
- Python 3.11
- Flask
- Pandas / NumPy
- Chart.js (frontend)
