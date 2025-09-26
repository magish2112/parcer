import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

def get_bybit_client(testnet=False):
    """
    Инициализация клиента Bybit с API-ключами из переменных окружения
    """
    API_KEY = os.getenv('BYBIT_API_KEY')
    API_SECRET = os.getenv('BYBIT_API_SECRET')
    return HTTP(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=testnet
    )