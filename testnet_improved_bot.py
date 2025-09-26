#!/usr/bin/env python3
"""
Улучшенный торговый бот для Bybit Testnet с новой моделью PPO
Использует рефакторингованную логику пирамидинга и выходов из позиций
"""

import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import logging
from datetime import datetime
import yaml

# Импортируем функции для работы с данными
import sys
sys.path.append('src')
try:
    from data.indicators import add_indicators
except ImportError:
    # Альтернативный импорт
    import src.data.indicators as indicators_module
    add_indicators = indicators_module.add_indicators

# Загружаем переменные окружения
load_dotenv()

# Загружаем конфигурацию (та же, что и при обучении)
with open('config/improved_trading_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Параметры торговли
SYMBOL = 'BTCUSDT'
QTY_BASE = 0.01  # УВЕЛИЧЕННЫЙ базовый размер позиции (0.01 BTC для ~1000 USDT)
INTERVAL = '240'  # 4H свечи (как при обучении!)
WINDOW_SIZE = 24  # Размер окна для модели
MODEL_PATH = 'new_improved_ppo_model.zip'

# Спецификации контракта BTCUSDT (будут получены динамически)
CONTRACT_SPECS = {
    'min_order_qty': 0.001,  # Минимальный размер контракта
    'qty_step': 0.001,       # Шаг размера контракта
}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('testnet_improved_log.txt'),
        logging.StreamHandler()
    ]
)

class BybitTestnetBot:
    """Улучшенный торговый бот для Bybit Testnet"""

    def __init__(self, model_path=None, testnet=True):
        self.model_path = model_path or MODEL_PATH
        self.testnet = testnet
        self.client = None
        self.model = None
        self.position_active = False
        self.entry_levels = []
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.total_invested = 0.0

        # Параметры пирамидинга (синхронизированы с новой логикой)
        self.pyramid_levels = [0.10, 0.20, 0.50]  # 10%, 20%, 50% от депозита
        self.pyramid_thresholds = [0.05, 0.08, 0.08]  # Просадки для активации
        self.current_level = 0
        self.drawdown_max = 0.0

        # Параметры выхода
        self.profit_target_single = 0.03  # 3% для первой позиции
        self.profit_target_multi = 0.05   # 5% для нескольких позиций

        self.setup_client()
        self.load_model()

    def setup_client(self):
        """Настройка клиента Bybit"""
        try:
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')

            if not api_key or not api_secret:
                raise ValueError("Не найдены API ключи BYBIT_API_KEY или BYBIT_API_SECRET")

            self.client = HTTP(
                api_key=api_key,
                api_secret=api_secret,
                testnet=self.testnet
            )
            logging.info("Клиент Bybit настроен успешно")
            print("Клиент Bybit настроен успешно")
        except Exception as e:
            logging.error(f"❌ Ошибка настройки клиента Bybit: {e}")
            raise

    def load_model(self):
        """Загрузка обученной модели PPO"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")

            self.model = PPO.load(self.model_path)
            logging.info(f"Модель загружена: {self.model_path}")
            print(f"✅ Модель загружена: {self.model_path}")
        except Exception as e:
            logging.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def get_contract_specs(self):
        """Получение спецификаций контракта BTCUSDT"""
        try:
            response = self.client.get_instruments_info(
                category="spot",
                symbol=SYMBOL
            )

            if response['retCode'] == 0:
                contract = response['result']['list'][0]

                # Отладка: показываем структуру ответа
                logging.info(f"Структура контракта: {list(contract.keys())}")

                # Проверяем наличие ключей в ответе
                min_order_qty = contract.get('minOrderQty') or contract.get('minQty') or '0.001'
                qty_step = contract.get('qtyStep') or contract.get('stepSize') or '0.001'

                logging.info(f"Найденные значения: minOrderQty={min_order_qty}, qtyStep={qty_step}")

                specs = {
                    'min_order_qty': float(min_order_qty),
                    'qty_step': float(qty_step),
                    'max_order_qty': float(contract.get('maxOrderQty', 100.0)),
                }

                logging.info(f"Спецификации контракта {SYMBOL}:")
                logging.info(f"   Минимальный размер: {specs['min_order_qty']}")
                logging.info(f"   Шаг размера: {specs['qty_step']}")
                logging.info(f"   Максимальный размер: {specs['max_order_qty']}")

                # Обновляем глобальные спецификации
                global CONTRACT_SPECS
                CONTRACT_SPECS.update(specs)

                return specs
            else:
                logging.error(f"Ошибка получения спецификаций контракта: {response['retMsg']}")
                logging.error(f"Полный ответ API: {response}")
                return CONTRACT_SPECS
        except Exception as e:
            logging.error(f"Ошибка получения спецификаций контракта: {e}")
            return CONTRACT_SPECS

    def round_qty_to_contract_specs(self, qty):
        """Округление количества до допустимых значений контракта"""
        try:
            min_qty = CONTRACT_SPECS['min_order_qty']
            qty_step = CONTRACT_SPECS['qty_step']

            # Если количество меньше минимума, используем минимум
            if qty < min_qty:
                return min_qty

            # Округляем до ближайшего шага
            rounded_qty = round(qty / qty_step) * qty_step

            # Убеждаемся, что результат не меньше минимума
            rounded_qty = max(rounded_qty, min_qty)

            return rounded_qty

        except Exception as e:
            logging.error(f"Ошибка округления количества: {e}")
            return max(qty, CONTRACT_SPECS['min_order_qty'])

    def save_trading_state(self):
        """Сохранение состояния торговли для восстановления после перезапуска"""
        try:
            state = {
                'position_active': self.position_active,
                'position_size': self.position_size,
                'avg_entry_price': self.avg_entry_price,
                'total_invested': self.total_invested,
                'entry_levels': self.entry_levels,
                'current_level': self.current_level,
                'drawdown_max': self.drawdown_max,
                'timestamp': datetime.now().isoformat()
            }

            import json
            with open('trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)

            logging.info("Состояние торговли сохранено в trading_state.json")
            print("💾 Состояние торговли сохранено")

        except Exception as e:
            logging.error(f"Ошибка сохранения состояния: {e}")

    def load_trading_state(self):
        """Загрузка состояния торговли при запуске"""
        try:
            import json
            if os.path.exists('trading_state.json'):
                with open('trading_state.json', 'r') as f:
                    state = json.load(f)

                # Восстанавливаем состояние
                self.position_active = state.get('position_active', False)
                self.position_size = state.get('position_size', 0.0)
                self.avg_entry_price = state.get('avg_entry_price', 0.0)
                self.total_invested = state.get('total_invested', 0.0)
                self.entry_levels = state.get('entry_levels', [])
                self.current_level = state.get('current_level', 0)
                self.drawdown_max = state.get('drawdown_max', 0.0)

                timestamp = state.get('timestamp', 'N/A')
                logging.info(f"Состояние торговли загружено из {timestamp}")
                print(f"📂 Состояние торговли загружено из {timestamp}")

                return True
            else:
                logging.info("Файл состояния не найден, начинаем с чистого листа")
                return False

        except Exception as e:
            logging.error(f"Ошибка загрузки состояния: {e}")
            return False

    def get_wallet_balance(self):
        """Получение баланса кошелька (USDT + BTC)"""
        try:
            balance = self.client.get_wallet_balance(
                accountType="UNIFIED",  # Для testnet нужен UNIFIED
                coin="USDT,BTC"
            )
            if balance['retCode'] == 0:
                coins = balance['result']['list'][0]['coin']

                usdt_balance = 0.0
                btc_balance = 0.0

                for coin in coins:
                    if coin['coin'] == 'USDT':
                        usdt_balance = float(coin['walletBalance'])
                    elif coin['coin'] == 'BTC':
                        btc_balance = float(coin['walletBalance'])

                return {
                    'usdt': usdt_balance,
                    'btc': btc_balance,
                    'total_usdt': usdt_balance + (btc_balance * 100000)  # Примерная оценка в USDT
                }
            else:
                logging.error(f"Ошибка получения баланса: {balance['retMsg']}")
                return None
        except Exception as e:
            logging.error(f"Ошибка проверки баланса: {e}")
            return None

    def get_market_data(self, limit=100):
        """Получение рыночных данных с Bybit"""
        try:
            response = self.client.get_kline(
                category="spot",
                symbol=SYMBOL,
                interval=INTERVAL,
                limit=limit
            )

            if response['retCode'] != 0:
                logging.error(f"Ошибка получения данных: {response['retMsg']}")
                return None

            # Преобразование в DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=[
                'start_time', 'open', 'high', 'low', 'close',
                'volume', 'turnover'
            ])

            # Конвертация типов
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            # Сортировка по времени
            df = df.sort_values('start_time').reset_index(drop=True)

            # Добавление технических индикаторов
            df = self.add_indicators(df)

            return df

        except Exception as e:
            logging.error(f"Ошибка получения рыночных данных: {e}")
            return None

    def add_indicators(self, df):
        """Добавление технических индикаторов (синхронизировано с обучением)"""
        try:
            # Используем тот же расчет индикаторов, что и при обучении
            df_with_indicators = add_indicators(df.copy(), config['data'])
            return df_with_indicators

        except Exception as e:
            logging.error(f"Ошибка добавления индикаторов: {e}")
            return df

    def prepare_observation(self, df):
        """Подготовка наблюдения для модели"""
        try:
            # Берем последние WINDOW_SIZE свечей
            recent_data = df.tail(WINDOW_SIZE)

            # Используем ТОТ ЖЕ набор признаков, что и при обучении (24 признака)
            features = config['data']['feature_list']

            # Получаем данные признаков
            obs_data = recent_data[features].values

            # Обработка NaN значений
            obs_data = np.nan_to_num(obs_data, nan=0.0)

            # Нормализация
            obs_mean = np.mean(obs_data, axis=0)
            obs_std = np.std(obs_data, axis=0)
            obs_std = np.where(obs_std == 0, 1, obs_std)  # Избегание деления на 0

            obs_normalized = (obs_data - obs_mean) / obs_std

            # Приведение к форме (WINDOW_SIZE, WINDOW_SIZE)
            window_size = WINDOW_SIZE
            n_features = obs_normalized.shape[1]

            # Отладка: показываем статистику нормализованных данных
            logging.info(f"Нормализованные данные: shape={obs_normalized.shape}, mean={obs_normalized.mean():.4f}, std={obs_normalized.std():.4f}")

            if n_features < window_size:
                # Добавляем нули справа
                padding = np.zeros((window_size, window_size - n_features))
                obs_matrix = np.column_stack([obs_normalized, padding])
            else:
                # Обрезаем до window_size
                obs_matrix = obs_normalized[:, :window_size]

            logging.info(f"Финальная матрица наблюдения: shape={obs_matrix.shape}, range=[{obs_matrix.min():.4f}, {obs_matrix.max():.4f}]")
            return obs_matrix.astype(np.float32)

        except Exception as e:
            logging.error(f"Ошибка подготовки наблюдения: {e}")
            return np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)

    def get_model_action(self, observation):
        """Получение действия от модели"""
        try:
            # Модель ожидает форму (batch_size, height, width)
            obs_expanded = np.expand_dims(observation, axis=0)

            # Отладка: показываем вход модели
            logging.info(f"Вход модели: shape={obs_expanded.shape}, mean={obs_expanded.mean():.4f}, std={obs_expanded.std():.4f}")

            # УБИРАЕМ deterministic=True для тестирования - модель должна пробовать разные действия
            action, _ = self.model.predict(obs_expanded, deterministic=False)

            logging.info(f"Действие модели: {action}")

            # Конвертация в int
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action.item())
                elif action.size > 0:
                    action = int(action[0])
                else:
                    action = 0
            else:
                action = int(action)

            return action

        except Exception as e:
            logging.error(f"Ошибка получения действия от модели: {e}")
            return 0  # ДЕРЖАТЬ по умолчанию

    def calculate_position_size(self, balance, price, drawdown):
        """Расчет размера позиции с учетом минимальных требований Bybit"""
        try:
            if self.current_level >= len(self.pyramid_levels):
                return 0.0  # Все уровни использованы

            # Проверяем, можем ли активировать следующий уровень
            threshold = self.pyramid_thresholds[self.current_level] if self.current_level > 0 else 0.0

            if self.current_level == 0 or drawdown >= threshold:
                # Рассчитываем размер позиции для текущего уровня
                level_percentage = self.pyramid_levels[self.current_level]

                # Для тестнета используем фиксированный минимальный размер для первой позиции
                logging.info(f"Текущий уровень: {self.current_level}, процент уровня: {level_percentage*100:.0f}%")
                logging.info(f"Баланс: {balance:.2f} USDT, цена: {price:.2f}")

                if self.current_level == 0:
                    # Для тестнета используем фиксированную сумму 1000 USDT для первой позиции
                    position_value_usdt = 1000.0  # Фиксированная сумма 1000 USDT
                    logging.info(f"Первая позиция (тестнет фикс): {position_value_usdt:.2f} USDT")
                else:
                    position_value_usdt = balance * level_percentage
                    logging.info(f"Последующая позиция: {position_value_usdt:.2f} USDT")

                position_size = position_value_usdt / price

                # Дополнительная проверка на минимальный размер
                position_size = max(position_size, 0.001)  # Минимум 0.001 BTC
                position_size = self.round_qty_to_contract_specs(position_size)
                position_value_usdt = position_size * price

                logging.info(f"Размер позиции округлен до: {position_size:.4f} BTC (мин. стоимость: {position_value_usdt:.2f} USDT)")

                logging.info(f"Уровень {self.current_level + 1}: {level_percentage*100:.0f}% депозита")
                return position_size
            else:
                return 0.0  # Условия для уровня не выполнены

        except Exception as e:
            logging.error(f"Ошибка расчета размера позиции: {e}")
            return 0.0

    def execute_buy(self, qty):
        """Выполнение покупки с подробным логированием"""
        try:
            logging.info(f"Попытка покупки: qty={qty}, symbol={SYMBOL}")
            print(f"🔄 Отправка ордера на покупку: {qty} BTC...")

            result = self.client.place_order(
                category="spot",
                symbol=SYMBOL,
                side="Buy",
                orderType="Market",
                qty=str(qty),  # Конвертируем в строку
                timeInForce="GoodTillCancel"
            )

            logging.info(f"Ответ API: {result}")

            if result['retCode'] == 0:
                order_id = result.get('result', {}).get('orderId', 'N/A')
                logging.info(f"Покупка выполнена успешно: orderId={order_id}, qty={qty} {SYMBOL}")
                print(f"Покупка выполнена: {qty} BTC (orderId: {order_id})")
                return True
            else:
                error_code = result.get('retCode', 'N/A')
                error_msg = result.get('retMsg', 'N/A')
                logging.error(f"Ошибка API покупки: code={error_code}, message={error_msg}")
                print(f"❌ Ошибка API: {error_msg} (код: {error_code})")
                return False

        except Exception as e:
            # Безопасное логирование без Unicode символов
            safe_error_msg = str(e).replace('→', '->').replace('→', '->')
            logging.error(f"Исключение при покупке: {safe_error_msg}")
            print(f"❌ Критическая ошибка при покупке: {safe_error_msg}")
            return False

    def execute_sell(self, qty):
        """Выполнение продажи"""
        try:
            result = self.client.place_order(
                category="spot",
                symbol=SYMBOL,
                side="Sell",
                orderType="Market",
                qty=str(qty),  # Конвертируем в строку
                timeInForce="GoodTillCancel"
            )

            if result['retCode'] == 0:
                logging.info(f"Продажа выполнена: {qty:.6f} {SYMBOL}")
                print(f"Продажа выполнена: {qty:.6f} {SYMBOL}")
                return True
            else:
                logging.error(f"Ошибка продажи: {result['retMsg']}")
                return False

        except Exception as e:
            logging.error(f"Ошибка выполнения продажи: {e}")
            return False

    def check_exit_conditions(self, current_price):
        """Проверка условий выхода из позиции"""
        try:
            if not self.position_active or self.position_size == 0:
                return False

            # Расчет текущей прибыли
            current_profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price

            # Условия выхода
            if self.current_level == 0:
                # Одна позиция - выход при 3% профита
                if current_profit_pct >= self.profit_target_single:
                    logging.info(f"🎯 Выход из позиции: профит {current_profit_pct:.2f}% (1 уровень)")
                    return True
            else:
                # Множественные позиции - выход при 5% профита
                if current_profit_pct >= self.profit_target_multi:
                    logging.info(f"🎯 Выход из позиции: профит {current_profit_pct:.2f}% (много уровней)")
                    return True

            return False

        except Exception as e:
            logging.error(f"Ошибка проверки условий выхода: {e}")
            return False

    def update_drawdown(self, current_price):
        """Обновление максимальной просадки"""
        try:
            if self.position_active and self.position_size > 0:
                current_profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
                if current_profit_pct < 0:
                    self.drawdown_max = max(self.drawdown_max, abs(current_profit_pct))
        except Exception as e:
            logging.error(f"Ошибка обновления просадки: {e}")

    def run(self):
        """Основной цикл торговли"""
        print("🚀 ЗАПУСК УЛУЧШЕННОГО ТОРГОВОГО БОТА BYBIT TESTNET")
        print("=" * 80)
        print("🎯 СПотовая торговля BTCUSDT на Bybit Testnet")
        print("🎯 Новая логика пирамидинга: 10% -> 20% -> 50%")
        print("💰 Выход в безубыток: 3% (1 уровень) / 5% (много уровней)")
        print("🛡️ Отключены ранние стоп-лоссы")
        print("=" * 80)

        # Получение спецификаций контракта
        print("📋 Получение спецификаций контракта...")
        contract_specs = self.get_contract_specs()

        # Проверка баланса
        balance_info = self.get_wallet_balance()
        if balance_info is None:
            print("❌ Не удалось получить баланс")
            return

        usdt_balance = balance_info['usdt']
        btc_balance = balance_info['btc']
        total_balance = balance_info['total_usdt']

        print(f"💰 Баланс:")
        print(f"   USDT: {usdt_balance:.2f}")
        print(f"   BTC: {btc_balance:.6f}")
        print(f"   Общий: ~{total_balance:.2f} USDT")
        print(f"📊 Контрактные спецификации:")
        print(f"   Мин. размер: {contract_specs['min_order_qty']}")
        print(f"   Шаг размера: {contract_specs['qty_step']}")

        # Пытаемся загрузить предыдущее состояние торговли
        state_loaded = self.load_trading_state()

        # Определяем состояние позиции
        if not state_loaded:
            # Если состояние не загружено, проверяем баланс
            if btc_balance > 0.0001:  # Есть BTC позиция
                self.position_active = True
                self.position_size = btc_balance
                # Получаем текущую цену для расчета средней цены входа
                df = self.get_market_data(limit=10)
                if df is not None:
                    current_price = df.iloc[-1]['close']
                    self.avg_entry_price = current_price  # Примерно, в реальности нужно хранить реальную цену входа
                    self.total_invested = btc_balance * self.avg_entry_price
                    print(f"📊 Найдена существующая позиция BTC: {btc_balance:.6f} на цене ~{self.avg_entry_price:.2f}")
                else:
                    print("⚠️  Не удалось получить текущую цену для оценки позиции")
                    return
            else:
                print("🎯 Поиск точки входа для новой позиции")
                self.position_active = False
        else:
            # Состояние загружено, проверяем соответствие с реальным балансом
            if abs(self.position_size - btc_balance) > 0.00001:
                logging.warning(f"Несоответствие размера позиции: сохранено {self.position_size:.6f}, реально {btc_balance:.6f}")
                # Корректируем по реальному балансу
                self.position_size = btc_balance
                if btc_balance > 0.0001:
                    self.position_active = True
                    if self.avg_entry_price > 0:
                        self.total_invested = btc_balance * (self.total_invested / self.position_size)
                else:
                    self.position_active = False

        # Используем USDT баланс для расчета новых позиций
        balance = usdt_balance
        if balance < 100:
            print("⚠️  Недостаточно USDT для новых сделок")
            # Но продолжаем, если есть BTC позиция для управления

        total_trades = 0
        successful_trades = 0
        iteration = 0
        start_time = time.time()
        max_runtime = 24 * 60 * 60  # Максимум 24 часа работы
        daily_stats_interval = 60  # Статистика каждый час

        try:
            while True:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')

                # Проверка времени работы
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime:
                    print(f"\n⏰ Время теста истекло ({max_runtime/60:.1f} минут)")
                    break

                print(f"\n🔄 Итерация #{iteration} - {current_time}")

                # Получение рыночных данных
                df = self.get_market_data(limit=100)
                if df is None or len(df) < WINDOW_SIZE:
                    print("❌ Не удалось получить рыночные данные")
                    time.sleep(60)
                    continue

                current_price = df.iloc[-1]['close']

                # Подготовка наблюдения для модели
                observation = self.prepare_observation(df)

                # Получение действия от модели
                action = self.get_model_action(observation)

                # Отладочная информация о действии модели
                action_names = {0: 'ДЕРЖАТЬ', 1: 'КУПИТЬ', 2: 'ПРОДАТЬ'}
                print(f"🤖 Модель решила: {action} ({action_names[action]})")
                logging.info(f"Модель решила: {action} ({action_names[action]})")

                # Показываем ключевые индикаторы
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    print(f"📊 RSI: {last_row.get('rsi', 'N/A'):.1f} | MACD: {last_row.get('macd', 'N/A'):.4f} | "
                          f"Объем: {last_row.get('volume', 0):.0f}")

                # Обновление просадки
                self.update_drawdown(current_price)

                # Проверка условий выхода
                if self.check_exit_conditions(current_price):
                    # Закрываем всю позицию
                    if self.execute_sell(self.position_size):
                        total_trades += 1
                        successful_trades += 1

                        # Расчет прибыли
                        profit_usdt = self.total_invested * (current_price - self.avg_entry_price) / self.avg_entry_price

                        print("🎉 ВЫХОД ИЗ ПОЗИЦИИ:")
                        print(f"   Цена входа: {self.avg_entry_price:.2f}")
                        print(f"   Цена выхода: {current_price:.2f}")
                        print(f"   Просадка макс: {self.drawdown_max:.1f}%")
                        print(f"   Прибыль: {profit_usdt:.2f} USDT")
                        print("=" * 50)

                    # Сброс состояния позиции
                    self.position_active = False
                    self.entry_levels = []
                    self.position_size = 0.0
                    self.avg_entry_price = 0.0
                    self.total_invested = 0.0
                    self.current_level = 0
                    self.drawdown_max = 0.0

                    continue

                # Логика принятия решений на основе состояния позиции
                if action == 1:  # Модель дает сигнал КУПИТЬ
                    if not self.position_active:  # Нет позиции - открываем новую
                        print("🟢 СИГНАЛ ПОКУПКИ ОТ МОДЕЛИ - открываем позицию")
                        logging.info("СИГНАЛ ПОКУПКИ ОТ МОДЕЛИ - начинаем процесс открытия позиции")

                        # Расчет размера позиции
                        qty = self.calculate_position_size(balance, current_price, self.drawdown_max)
                        logging.info(f"Расчет размера позиции: {qty:.6f} BTC при цене {current_price:.2f}")

                        if qty > 0:
                            # Округляем qty согласно спецификациям контракта
                            qty_rounded = self.round_qty_to_contract_specs(qty)
                            logging.info(f"Округленный размер позиции: {qty_rounded:.4f} BTC")
                            print(f"📊 Рассчитан размер позиции: {qty:.6f} BTC -> {qty_rounded:.4f} BTC (согласно контракту)")

                            # Расчет стоимости с комиссией
                            position_cost = qty_rounded * current_price
                            commission_cost = position_cost * 0.001  # 0.1% комиссия
                            total_cost = position_cost + commission_cost

                            print(f"💰 Стоимость позиции: {position_cost:.2f} USDT")
                            print(f"💰 Комиссия входа: {commission_cost:.2f} USDT")
                            print(f"💰 Итого: {total_cost:.2f} USDT")

                            # Проверяем, что хватает средств
                            if total_cost > balance:
                                logging.warning(f"Недостаточно средств: {total_cost:.2f} > {balance:.2f}")
                                print(f"⚠️  Недостаточно средств для позиции: {total_cost:.2f} > {balance:.2f}")
                            elif qty_rounded < CONTRACT_SPECS['min_order_qty']:
                                logging.error(f"Размер позиции {qty_rounded:.4f} меньше минимального {CONTRACT_SPECS['min_order_qty']}")
                                print(f"⚠️  Размер позиции слишком маленький: {qty_rounded:.4f} < {CONTRACT_SPECS['min_order_qty']}")
                            else:
                                print(f"🔄 Выполняем покупку {qty_rounded:.4f} BTC...")
                                if self.execute_buy(qty_rounded):
                                    # Обновление состояния
                                    self.position_active = True
                                    self.entry_levels.append(current_price)
                                    self.position_size = qty_rounded  # Используем округленный размер
                                    self.avg_entry_price = current_price
                                    self.total_invested = total_cost  # Включая комиссию
                                    total_trades += 1
                                    successful_trades += 1

                                    logging.info(f"ПОЗИЦИЯ ОТКРЫТА: {qty_rounded:.4f} BTC по цене {current_price:.2f}")
                                    print("ПОЗИЦИЯ ОТКРЫТА:")
                                    print(f"   Цена входа: {current_price:.2f}")
                                    print(f"   Размер: {qty_rounded:.4f} BTC")
                                    print(f"   Инвестировано: {self.total_invested:.2f} USDT (с комиссией)")
                                    print("=" * 40)
                                else:
                                    logging.error("Ошибка выполнения покупки")
                                    print("❌ Ошибка открытия позиции")
                        else:
                            logging.info("Недостаточно средств для открытия позиции")
                            print("⏸️  Недостаточно средств или условия пирамидинга не выполнены")
                    else:  # Есть позиция - игнорируем сигнал покупки
                        logging.info("Сигнал покупки получен, но позиция уже активна")
                        print("🟢 Сигнал покупки, но позиция уже открыта - игнорируем")

                elif action == 2:  # Модель дает сигнал ПРОДАТЬ
                    if self.position_active:  # Есть позиция - закрываем
                        print("🔴 СИГНАЛ ПРОДАЖИ ОТ МОДЕЛИ - закрываем позицию")

                        # Расчет потенциальной прибыли с учетом комиссий
                        commission_rate = 0.001  # 0.1% комиссия
                        potential_profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
                        commission_cost = self.total_invested * commission_rate * 2  # Вход + выход
                        net_profit_usdt = self.total_invested * potential_profit_pct - commission_cost
                        net_profit_pct = net_profit_usdt / self.total_invested

                        print(f"📊 Анализ продажи:")
                        print(f"   Текущая прибыль: {potential_profit_pct:.4f}%")
                        print(f"   Комиссия: {commission_cost:.2f} USDT")
                        print(f"   Чистая прибыль: {net_profit_pct:.4f}% ({net_profit_usdt:.2f} USDT)")

                        # Минимальный порог прибыли 0.5% после комиссий
                        min_profit_threshold = 0.005  # 0.5%

                        if net_profit_pct >= min_profit_threshold:
                            print(f"✅ Продажа выгодна (прибыль > {min_profit_threshold*100:.1f}%)")
                            # Округляем размер позиции для продажи
                            sell_qty = min(self.round_qty_to_contract_specs(self.position_size),
                                         self.round_qty_to_contract_specs(balance_info['btc']))

                            if sell_qty != self.position_size:
                                print(f"📊 Округление размера продажи: {self.position_size:.6f} -> {sell_qty:.4f}")

                            if self.execute_sell(sell_qty):
                                total_trades += 1
                                successful_trades += 1

                                print("ПОЗИЦИЯ ЗАКРЫТА ПО СИГНАЛУ:")
                                print(f"   Цена входа: {self.avg_entry_price:.2f}")
                                print(f"   Цена выхода: {current_price:.2f}")
                                print(f"   Валовая прибыль: {potential_profit_pct:.4f}%")
                                print(f"   Комиссия: {commission_cost:.2f} USDT")
                                print(f"   Чистая прибыль: {net_profit_pct:.4f}% ({net_profit_usdt:.2f} USDT)")
                                print("=" * 40)

                                # Сброс состояния
                                self.position_active = False
                                self.entry_levels = []
                                self.position_size = 0.0
                                self.avg_entry_price = 0.0
                                self.total_invested = 0.0
                                self.current_level = 0
                                self.drawdown_max = 0.0
                            else:
                                print("❌ Ошибка закрытия позиции")
                        else:
                            print(f"⏸️ Продажа невыгодна (прибыль < {min_profit_threshold*100:.1f}%) - ждем лучших условий")
                            logging.info(f"Продажа отменена: чистая прибыль {net_profit_pct:.4f}% < {min_profit_threshold*100:.1f}%")
                    else:  # Нет позиции - игнорируем сигнал продажи
                        print("🔴 Сигнал продажи, но позиции нет - игнорируем")

                else:  # Сигнал ДЕРЖАТЬ
                    print("⏸️  ДЕРЖАТЬ позицию")

                    if self.position_active:
                        current_profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
                        print(f"   Текущий профит: {current_profit_pct:.2f}%")
                        print(f"   Просадка макс: {self.drawdown_max:.1f}%")
                        print(f"   Размер позиции: {self.position_size:.4f} BTC")
                # Статистика каждый час
                elapsed_time = time.time() - start_time
                if iteration % daily_stats_interval == 0 or iteration == 1:
                    hours = elapsed_time // 3600
                    minutes = (elapsed_time % 3600) // 60

                    print(f"\n📊 СТАТИСТИКА ({int(hours)}ч {int(minutes)}м):")
                    print(f"   Итераций: {iteration}")
                    print(f"   Всего сделок: {total_trades}")
                    print(f"   Успешных: {successful_trades}")
                    if total_trades > 0:
                        success_rate = (successful_trades / total_trades) * 100
                        print(f"   Успешность: {success_rate:.1f}%")
                        # Показываем общую прибыль
                        print(f"   Всего прибыль: ~{total_trades * 5:.2f} USDT (оценка)")

                    print(f"   Текущая цена: {current_price:.2f}")
                    if self.position_active:
                        current_profit_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
                        print(f"   Позиция: {self.position_size:.4f} BTC")
                        print(f"   Средняя цена входа: {self.avg_entry_price:.2f}")
                        print(f"   Текущий P&L: {current_profit_pct:.2f}%")
                        print(f"   Просадка макс: {self.drawdown_max:.1f}%")

                    # Проверяем баланс каждый час
                    if iteration % daily_stats_interval == 0:
                        current_balance = self.get_wallet_balance()
                        if current_balance:
                            print(f"   Баланс USDT: {current_balance['usdt']:.2f}")
                            print(f"   Баланс BTC: {current_balance['btc']:.6f}")

                    print("=" * 80)

                # Ожидание между итерациями
                print("⏰ Ожидание 1 минуты...")
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n⏹️  Остановка бота по команде пользователя")
            self.save_trading_state()
        except Exception as e:
            logging.error(f"Критическая ошибка в основном цикле: {e}")
            print(f"❌ Критическая ошибка: {e}")
            self.save_trading_state()

        # Финальная статистика
        elapsed_time = time.time() - start_time
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60

        print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
        print("=" * 80)
        print(f"   Время работы: {int(hours)}ч {int(minutes)}м")
        print(f"   Итераций: {iteration}")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Успешных: {successful_trades}")
        if total_trades > 0:
            success_rate = (successful_trades / total_trades) * 100
            print(f"   Успешность: {success_rate:.1f}%")
        print(f"   Финальная цена: {current_price:.2f}")
        print("=" * 80)
        print("📁 Логи сохранены в: testnet_improved_log.txt")


def main():
    """Точка входа"""
    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        print("Сначала обучите модель с помощью: python main_improved.py --mode train")
        return

    # Запуск бота
    bot = BybitTestnetBot(model_path=MODEL_PATH)
    bot.run()


if __name__ == '__main__':
    main()
