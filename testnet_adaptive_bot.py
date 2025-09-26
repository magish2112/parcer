#!/usr/bin/env python3
"""
Адаптивный торговый бот для Bybit Testnet с обучением на исторических данных
"""

import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from api import get_bybit_client
from data import get_ohlcv, add_indicators
from model import get_action
from trade import market_buy, market_sell
import logging
from config import SYMBOL, DATASET_PATH
from env_crypto import CustomCryptoTradingEnv
from sklearn.preprocessing import StandardScaler
import joblib

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', 
                   filename='testnet_adaptive_log.txt')

# Параметры торговли
SYMBOL = 'BTCUSDT'
QTY = 0.1  # Размер позиции для депозита 200,000+ USDT
INTERVAL = '15'  # 15-минутные свечи
LIMIT = 500  # Больше данных для обучения
MODEL_PATH = 'ppo_crypto_model_best_improved.zip'

# Список признаков
FEATURE_LIST = [
    'close', 'volume', 'rsi', 'ema_14', 'macd', 'atr', 'bb_bbm', 'adx', 'cci', 'roc', 'stoch',
    'crsi', 'sideways_volume', 'dist_to_support', 'dist_to_resistance', 'false_breakout', 
    'volume_spike', 'trend_ema', 'support', 'resistance', 'ema200_d1', 'atr_mult_trailing_stop'
]

# Параметры среды
WINDOW_SIZE = 24
SL_PCT = 0.03
TP_PCT = 0.07

# Параметры обучения
LEARNING_EPISODES = 1000  # Количество эпизодов для обучения на исторических данных
MIN_DATA_POINTS = 1000    # Минимальное количество точек данных для обучения

def get_ohlcv_from_bybit(client, symbol='BTCUSDT', interval='15', limit=200):
    """Получает OHLCV данные с индикаторами прямо с Bybit"""
    try:
        # Получаем kline данные с Bybit
        response = client.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if response['retCode'] != 0:
            print(f"❌ Ошибка получения данных: {response['retMsg']}")
            return None
            
        # Преобразуем в DataFrame
        klines = response['result']['list']
        df = pd.DataFrame(klines, columns=[
            'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Конвертируем типы данных
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['turnover'] = df['turnover'].astype(float)
        
        # Сортируем по времени (от старых к новым)
        df = df.sort_values('start_time').reset_index(drop=True)
        
        # Добавляем индикаторы
        df = add_indicators(df)
        
        print(f"✅ Получено {len(df)} свечей с индикаторами с Bybit")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка получения данных с Bybit: {e}")
        return None

def create_training_data(df):
    """Создает обучающие данные с метками на основе будущих движений цены"""
    print("📊 Создание обучающих данных...")
    
    # Добавляем будущие цены для создания меток
    df['future_price_1'] = df['close'].shift(-1)  # Цена через 1 период
    df['future_price_3'] = df['close'].shift(-3)  # Цена через 3 периода
    df['future_price_5'] = df['close'].shift(-5)  # Цена через 5 периодов
    df['future_price_10'] = df['close'].shift(-10)  # Цена через 10 периодов
    
    # Создаем метки на основе будущих движений
    # 0 = ДЕРЖАТЬ, 1 = КУПИТЬ, 2 = ПРОДАТЬ
    labels = []
    
    for i in range(len(df) - 10):  # Оставляем 10 последних точек
        current_price = df.iloc[i]['close']
        future_1 = df.iloc[i]['future_price_1']
        future_3 = df.iloc[i]['future_price_3']
        future_5 = df.iloc[i]['future_price_5']
        future_10 = df.iloc[i]['future_price_10']
        
        # Вычисляем процентные изменения
        change_1 = (future_1 - current_price) / current_price if not pd.isna(future_1) else 0
        change_3 = (future_3 - current_price) / current_price if not pd.isna(future_3) else 0
        change_5 = (future_5 - current_price) / current_price if not pd.isna(future_5) else 0
        change_10 = (future_10 - current_price) / current_price if not pd.isna(future_10) else 0
        
        # Взвешенное изменение (больше веса ближайшим периодам)
        weighted_change = (change_1 * 0.4 + change_3 * 0.3 + change_5 * 0.2 + change_10 * 0.1)
        
        # Получаем текущие индикаторы
        rsi = df.iloc[i]['rsi'] if not pd.isna(df.iloc[i]['rsi']) else 50
        macd = df.iloc[i]['macd'] if not pd.isna(df.iloc[i]['macd']) else 0
        volume = df.iloc[i]['volume'] if not pd.isna(df.iloc[i]['volume']) else 0
        avg_volume = df['volume'].rolling(20).mean().iloc[i] if not pd.isna(df['volume'].rolling(20).mean().iloc[i]) else volume
        ema_14 = df.iloc[i]['ema_14'] if not pd.isna(df.iloc[i]['ema_14']) else current_price
        atr = df.iloc[i]['atr'] if not pd.isna(df.iloc[i]['atr']) else current_price * 0.01
        
        # Анализ тренда
        price_above_ema = current_price > ema_14
        macd_positive = macd > 0
        volume_spike = volume > avg_volume * 1.5
        
        # Создаем метки на основе комплексного анализа
        buy_signal = (
            weighted_change > 0.015 and  # Минимум 1.5% роста
            rsi < 75 and  # RSI не перекуплен
            price_above_ema and  # Цена выше EMA
            (macd_positive or volume_spike)  # MACD положительный или всплеск объема
        )
        
        sell_signal = (
            weighted_change < -0.015 and  # Минимум 1.5% падения
            rsi > 25 and  # RSI не перепродан
            (not price_above_ema or rsi > 70)  # Цена ниже EMA или RSI перекуплен
        )
        
        # Принимаем решение
        if buy_signal and not sell_signal:
            labels.append(1)  # КУПИТЬ
        elif sell_signal and not buy_signal:
            labels.append(2)  # ПРОДАТЬ
        else:
            labels.append(0)  # ДЕРЖАТЬ
    
    # Добавляем метки в DataFrame
    df['label'] = [0] * (len(df) - len(labels)) + labels
    
    print(f"✅ Создано {len(labels)} обучающих примеров")
    print(f"   - Покупки: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
    print(f"   - Продажи: {labels.count(2)} ({labels.count(2)/len(labels)*100:.1f}%)")
    print(f"   - Удержания: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
    
    return df

def train_model_on_historical_data(df, model_path=None):
    """Обучает модель на исторических данных"""
    print("🤖 Обучение модели на исторических данных...")
    
    # Создаем обучающие данные
    df_train = create_training_data(df.copy())
    
    # Подготавливаем данные для обучения
    train_data = df_train.dropna()
    
    if len(train_data) < MIN_DATA_POINTS:
        print(f"❌ Недостаточно данных для обучения: {len(train_data)} < {MIN_DATA_POINTS}")
        return None
    
    # Создаем признаки и метки
    X = train_data[FEATURE_LIST].values
    y = train_data['label'].values
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Создаем среду для обучения с временным файлом
    temp_data_path = 'temp_training_data.csv'
    train_data.to_csv(temp_data_path, index=False)
    
    env = CustomCryptoTradingEnv(
        df_path=temp_data_path,
        window_size=WINDOW_SIZE,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        symbol=SYMBOL,
        use_trend_filter=False,
        rf_enabled=False,
        log_trades_path='adaptive_training_trades.csv'
    )
    
    env.reset()
    
    # Создаем или загружаем модель
    if model_path and os.path.exists(model_path):
        print(f"📥 Загружаем существующую модель: {model_path}")
        model = PPO.load(model_path)
    else:
        print("🆕 Создаем новую модель")
        model = PPO('MlpPolicy', env, verbose=1)
    
    # Обучаем модель
    print(f"🎯 Начинаем обучение на {LEARNING_EPISODES} эпизодах...")
    model.learn(total_timesteps=LEARNING_EPISODES * 100)
    
    # Сохраняем обученную модель
    adaptive_model_path = 'ppo_adaptive_model.zip'
    model.save(adaptive_model_path)
    print(f"💾 Модель сохранена: {adaptive_model_path}")
    
    # Сохраняем нормализатор
    scaler_path = 'adaptive_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"💾 Нормализатор сохранен: {scaler_path}")
    
    # Удаляем временный файл
    if os.path.exists(temp_data_path):
        os.remove(temp_data_path)
        print(f"🗑️  Временный файл удален: {temp_data_path}")
    
    return model, scaler

def train_new_model(client):
    """Обучает новую модель на исторических данных"""
    print("\n📡 Загрузка исторических данных для обучения...")
    try:
        # Сначала пробуем загрузить локальный датасет
        df_historical = pd.read_csv(DATASET_PATH)
        print(f"📊 Загружен локальный датасет: {DATASET_PATH}")
        print(f"📊 Размер датасета: {len(df_historical)} строк")
        
        # Проверяем, есть ли уже индикаторы
        if 'rsi' not in df_historical.columns:
            print("🔧 Добавление технических индикаторов...")
            df_historical = add_indicators(df_historical)
        
        # Обучаем модель на исторических данных
        model, scaler = train_model_on_historical_data(df_historical)
        
    except Exception as e:
        print(f"❌ Ошибка загрузки локального датасета: {e}")
        print("📡 Получение данных с Bybit...")
        df_historical = get_ohlcv_from_bybit(client, symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
        if df_historical is None:
            print("❌ Не удалось получить данные с Bybit")
            return None, None
        print(f"📊 Получено {len(df_historical)} исторических точек данных")
        model, scaler = train_model_on_historical_data(df_historical)
    
    return model, scaler

def make_env():
    """Создает среду для торговли"""
    return CustomCryptoTradingEnv(
        df_path=DATASET_PATH,
        window_size=WINDOW_SIZE,
        sl_pct=SL_PCT,
        tp_pct=TP_PCT,
        symbol=SYMBOL,
        use_trend_filter=False,
        rf_enabled=False,
        log_trades_path='testnet_adaptive_trades.csv'
    )

def check_wallet_balance(client):
    """Проверяет баланс кошелька"""
    try:
        balance = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if balance['retCode'] == 0:
            usdt_balance = float(balance['result']['list'][0]['coin'][0]['walletBalance'])
            return usdt_balance
        else:
            print(f"❌ Ошибка получения баланса: {balance['retMsg']}")
            return None
    except Exception as e:
        print(f"❌ Ошибка проверки баланса: {e}")
        return None

def main():
    """Основная функция торгового бота"""
    print("🤖 АДАПТИВНЫЙ ТОРГОВЫЙ БОТ ДЛЯ BYBIT TESTNET")
    print("=" * 70)
    print("🧠 Обучение на исторических данных + реальная торговля")
    print("📊 Анализ паттернов роста/падения и объемов")
    print("🔒 Testnet - реальные деньги НЕ используются")
    print("=" * 70)
    
    # Настройка клиента Bybit для testnet
    client = get_bybit_client(testnet=True)
    
    # Проверяем подключение и баланс
    print("🔍 Проверка подключения и баланса...")
    usdt_balance = check_wallet_balance(client)
    if usdt_balance is None:
        print("❌ Не удалось получить баланс. Проверьте API ключи.")
        return
    
    print(f"✅ Подключение к testnet успешно!")
    print(f"💰 Баланс UNIFIED USDT: {usdt_balance}")
    
    simulation_mode = False
    if usdt_balance == 0:
        print("⚠️  В UNIFIED аккаунте нет средств!")
        print("💡 Запускаем режим симуляции с обучением")
        usdt_balance = 10000  # Виртуальный баланс для симуляции
        simulation_mode = True
    else:
        print(f"✅ Достаточно средств для торговли: {usdt_balance} USDT")
    
    # Проверяем, есть ли уже обученная модель
    adaptive_model_path = 'ppo_adaptive_model.zip'
    scaler_path = 'adaptive_scaler.pkl'
    
    if os.path.exists(adaptive_model_path) and os.path.exists(scaler_path):
        print(f"📥 Загружаем уже обученную модель: {adaptive_model_path}")
        try:
            model = PPO.load(adaptive_model_path)
            scaler = joblib.load(scaler_path)
            print("✅ Модель и нормализатор загружены успешно!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("🔄 Переходим к обучению новой модели...")
            model, scaler = train_new_model(client)
    else:
        print("🆕 Обученная модель не найдена. Обучаем новую модель...")
        model, scaler = train_new_model(client)
    
    if model is None:
        print("❌ Не удалось обучить модель. Остановка бота.")
        return
    
    # Создаем среду для торговли
    env = make_env()
    obs, info = env.reset()
    
    print(f"\n📊 Среда создана: {env.df.shape[0]:,} строк данных")
    print(f"🎯 Начальный баланс: {env.balance}")
    
    total_trades = 0
    successful_trades = 0
    total_profit = 0.0
    
    print(f"\n🚀 Адаптивный торговый бот запущен!")
    if simulation_mode:
        print("🎮 РЕЖИМ СИМУЛЯЦИИ - реальные сделки не выполняются")
    print("Для остановки нажмите Ctrl+C")
    print("=" * 70)
    
    iteration = 0
    start_time = time.time()
    
    while True:
        try:
            iteration += 1
            print(f"\n🔄 Итерация #{iteration} - {time.strftime('%H:%M:%S')}")
            
            # Получаем актуальные данные с Bybit
            print("📡 Получение актуальных данных с Bybit...")
            df_current = get_ohlcv_from_bybit(client, symbol=SYMBOL, interval=INTERVAL, limit=LIMIT)
            
            if df_current is None:
                print("❌ Не удалось получить данные с Bybit, пропускаем итерацию")
                time.sleep(60)
                continue
            
            # Подготавливаем состояние для модели
            state = df_current.tail(WINDOW_SIZE)[FEATURE_LIST].values
            state_scaled = scaler.transform(state)
            state_scaled = np.expand_dims(state_scaled, axis=0)
            
            # Получаем действие от обученной модели
            action = get_action(model, state_scaled)
            
            # Отладочная информация
            print(f"🔍 Анализ:")
            print(f"   - Цена: {df_current.iloc[-1]['close']:.2f}")
            print(f"   - RSI: {df_current.iloc[-1]['rsi']:.2f}")
            print(f"   - MACD: {df_current.iloc[-1]['macd']:.4f}")
            print(f"   - Объем: {df_current.iloc[-1]['volume']:.0f}")
            print(f"   - Действие: {action} ({'ДЕРЖАТЬ' if action == 0 else 'КУПИТЬ' if action == 1 else 'ПРОДАТЬ'})")
            
            # Выполняем действие в среде
            obs, reward, done, _, info = env.step([action])
            
            # Получаем информацию о состоянии
            balance = info.get('balance', None)
            position_size = info.get('position_size', None)
            avg_entry_price = info.get('avg_entry_price', None)
            drawdown = info.get('drawdown', 0)
            trade_count = info.get('trade_count', 0)
            total_profit = info.get('total_profit', 0)
            
            # Выводим текущее состояние
            print(f'📊 Баланс: {balance:.4f} | Позиция: {position_size:.6f} | '
                  f'Вход: {avg_entry_price:.2f} | Просадка: {drawdown*100:.1f}% | '
                  f'Сделка #{trade_count} | Прибыль: {total_profit:.4f}')
            
            # Выполняем реальные сделки на testnet (если не в режиме симуляции)
            if not simulation_mode:
                if action == 1:  # Покупка
                    logging.info('Сигнал: КУПИТЬ')
                    print('🟢 Сигнал: КУПИТЬ')
                    result = market_buy(client, SYMBOL, QTY)
                    if result:
                        total_trades += 1
                        successful_trades += 1
                        print(f'✅ Успешная покупка! Всего сделок: {total_trades}')
                    else:
                        print('❌ Ошибка покупки')
                        
                elif action == 2:  # Продажа
                    logging.info('Сигнал: ПРОДАТЬ')
                    print('🔴 Сигнал: ПРОДАТЬ')
                    result = market_sell(client, SYMBOL, QTY)
                    if result:
                        total_trades += 1
                        successful_trades += 1
                        print(f'✅ Успешная продажа! Всего сделок: {total_trades}')
                    else:
                        print('❌ Ошибка продажи')
                        
                else:  # Удержание
                    logging.info('Сигнал: ДЕРЖАТЬ')
                    print('⏸️  Сигнал: ДЕРЖАТЬ')
            else:
                # Режим симуляции
                action_names = {0: 'ДЕРЖАТЬ', 1: 'КУПИТЬ', 2: 'ПРОДАТЬ'}
                print(f'🎮 Симуляция: {action_names[action]}')
                if action in [1, 2]:
                    total_trades += 1
                    successful_trades += 1
            
            # Показываем статистику каждые 5 итераций
            if iteration % 5 == 0:
                elapsed_time = time.time() - start_time
                hours = elapsed_time // 3600
                minutes = (elapsed_time % 3600) // 60
                
                print(f"\n📈 СТАТИСТИКА (время работы: {int(hours)}ч {int(minutes)}м):")
                print(f"   Итераций: {iteration}")
                print(f"   Всего сделок: {total_trades}")
                print(f"   Успешных: {successful_trades}")
                if total_trades > 0:
                    success_rate = (successful_trades / total_trades) * 100
                    print(f"   Успешность: {success_rate:.1f}%")
                print(f"   Текущий баланс: {balance:.4f}")
                print(f"   Общая прибыль: {total_profit:.4f}")
                print("=" * 70)
            
            # Ждем 1 минуту между итерациями (для тестирования)
            print(f"⏰ Ожидание 1 минуты до следующей итерации...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n⏹️  Остановка бота...")
            break
        except Exception as e:
            logging.error(f'Ошибка в торговом цикле: {e}')
            print(f'❌ Ошибка: {e}')
            time.sleep(60)  # Ждем минуту при ошибке
    
    # Финальная статистика
    elapsed_time = time.time() - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    
    print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
    print("=" * 70)
    print(f"   Время работы: {int(hours)}ч {int(minutes)}м")
    print(f"   Итераций: {iteration}")
    print(f"   Всего сделок: {total_trades}")
    print(f"   Успешных: {successful_trades}")
    if total_trades > 0:
        success_rate = (successful_trades / total_trades) * 100
        print(f"   Успешность: {success_rate:.1f}%")
    print(f"   Финальный баланс: {balance:.4f}")
    print(f"   Общая прибыль: {total_profit:.4f}")
    print("=" * 70)
    print("📁 Логи сохранены в:")
    print("   - testnet_adaptive_log.txt")
    print("   - testnet_adaptive_trades.csv")

if __name__ == '__main__':
    main()
