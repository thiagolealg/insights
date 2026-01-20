import MetaTrader5 as mt5
import time
from colorama import init, Fore, Style

init()

if not mt5.initialize():
    print("Init failed")
    quit()

info = mt5.terminal_info()

if info.trade_allowed:
    print(f"{Fore.GREEN}ALGO TRADING IS ENABLED! ✅{Style.RESET_ALL}")
    # Retry Test Order
    print("Retrying Test Order...")
    
    symbol = "WING26"
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        price = tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 1.0,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price - 100,
            "tp": price + 100,
            "deviation": 20,
            "magic": 123456,
            "comment": "Cortex Retry",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{Fore.GREEN}ORDER SENT SUCCESS! Ticket: {res.order}{Style.RESET_ALL}")
        elif res.retcode == 10027:
             print(f"{Fore.RED}STILL BLOCKED (10027). Please enable Algo Trading button.{Style.RESET_ALL}")
        else:
             print(f"{Fore.RED}Failed: {res.retcode} - {res.comment}{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}ALGO TRADING IS DISABLED ❌{Style.RESET_ALL}")
    print("I cannot click the button for you (Security Restriction).")
    print("Please press 'Ctrl + E' in the MT5 Terminal.")

mt5.shutdown()
