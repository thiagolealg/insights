import MetaTrader5 as mt5
import time
from datetime import datetime

# Initialize MT5
if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()

symbol = "WING26"
print(f"Sending Test Order to {symbol}...")

# Ensure Symbol Selected
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}")
    mt5.shutdown()
    quit()

# Get Pice
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    print(f"Failed to get tick for {symbol}")
    mt5.shutdown()
    quit()

price = tick.ask
variation = 50.0

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 1.0,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": price - variation,
    "tp": price + variation,
    "deviation": 20,
    "magic": 999999,
    "comment": "Cortex Test Order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_RETURN,
}

print(f"Request: Buy 1.0 {symbol} @ {price} (TP:{price+variation} SL:{price-variation})")

result = mt5.order_send(request)

if result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Order Failed: {result.retcode}")
    print(f"Result: {result}")
else:
    print(f"Order SUCCESS! Ticket: {result.order}")

mt5.shutdown()
