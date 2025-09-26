from config import SYMBOL

def market_buy(client, symbol, qty):
    """
    Совершить рыночную покупку (Buy) на Bybit через pybit
    """
    try:
        result = client.place_order(
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty,
            timeInForce="GoodTillCancel"
        )
        print(f"[TRADE] BUY {qty} {symbol}: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] Не удалось купить: {e}")
        return None

def market_sell(client, symbol, qty):
    """
    Совершить рыночную продажу (Sell) на Bybit через pybit
    """
    try:
        result = client.place_order(
            category="linear",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=qty,
            timeInForce="GoodTillCancel"
        )
        print(f"[TRADE] SELL {qty} {symbol}: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] Не удалось продать: {e}")
        return None 