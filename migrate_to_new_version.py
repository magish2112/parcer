#!/usr/bin/env python3
"""
Скрипт для миграции со старой версии на новую архитектуру
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import yaml


def backup_old_files():
    """Создание резервной копии старых файлов"""
    print("📦 Создание резервной копии старых файлов...")
    
    backup_dir = Path("backup_old_version")
    backup_dir.mkdir(exist_ok=True)
    
    # Файлы для резервного копирования
    old_files = [
        "api.py", "config.py", "data.py", "model.py", "env_crypto.py",
        "trade.py", "train_adaptive_model.py", "testnet_adaptive_bot.py",
        "requirements.txt", "README.md"
    ]
    
    for file in old_files:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir / file)
            print(f"  ✅ {file} -> backup_old_version/{file}")
    
    print(f"📁 Резервная копия создана в: {backup_dir}")


def migrate_data_files():
    """Миграция файлов данных"""
    print("📊 Миграция файлов данных...")
    
    # Создаем директорию для данных
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Файлы данных для перемещения
    data_files = [
        "btc_4h_full_fixed.csv",
        "adaptive_training_trades.csv",
        "temp_training_data.csv",
        "adaptive_scaler.pkl",
        "ppo_adaptive_model.zip"
    ]
    
    for file in data_files:
        if os.path.exists(file):
            shutil.move(file, data_dir / file)
            print(f"  ✅ {file} -> data/{file}")
    
    print("📁 Файлы данных перемещены в директорию data/")


def create_env_file():
    """Создание файла .env из старых переменных"""
    print("🔐 Создание файла .env...")
    
    env_content = """# API ключи для Bybit
# Получите их на https://www.bybit.com/app/user/api-management
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Дополнительные настройки
TRADING_MODE=testnet  # testnet или mainnet
LOG_LEVEL=INFO
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("  ✅ Создан файл .env")
    print("  ⚠️  Не забудьте добавить ваши API ключи!")


def update_config_for_old_data():
    """Обновление конфигурации для работы со старыми данными"""
    print("⚙️  Обновление конфигурации...")
    
    # Загружаем конфигурацию
    config_path = "config/trading_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Обновляем пути к данным
        if os.path.exists("data/btc_4h_full_fixed.csv"):
            config['data']['dataset_path'] = 'data/btc_4h_full_fixed.csv'
        
        # Сохраняем обновленную конфигурацию
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print("  ✅ Конфигурация обновлена")
    else:
        print("  ⚠️  Файл конфигурации не найден")


def create_migration_script():
    """Создание скрипта для запуска старой модели в новой архитектуре"""
    print("🔄 Создание скрипта миграции...")
    
    migration_script = '''#!/usr/bin/env python3
"""
Скрипт для запуска старой модели в новой архитектуре
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import get_config
from src.utils.logging import setup_logging
from src.data.data_loader import DataLoader
from src.env.trading_env import OptimizedTradingEnv
from src.models.ppo_model import PPOModel


def run_old_model():
    """Запуск старой модели в новой архитектуре"""
    
    # Загружаем конфигурацию
    config = get_config('config/trading_config.yaml')
    
    # Настраиваем логирование
    logger = setup_logging(config.logging, "migration")
    logger.info("🔄 Запуск старой модели в новой архитектуре")
    
    try:
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data()
        
        # Создаем среду
        env = OptimizedTradingEnv(
            df_path=config.data.dataset_path,
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='migrated_trades.csv'
        )
        
        # Загружаем старую модель
        old_model_path = "data/ppo_adaptive_model.zip"
        if os.path.exists(old_model_path):
            model_config = config.model.__dict__.copy()
            model_config['model_path'] = old_model_path
            model = PPOModel(model_config)
            model.load(old_model_path)
            logger.info(f"✅ Старая модель загружена из {old_model_path}")
        else:
            logger.error(f"❌ Старая модель не найдена: {old_model_path}")
            return
        
        # Запускаем тестирование
        obs, info = env.reset()
        logger.info(f"🚀 Начальное состояние: Balance={info['balance']:.2f}")
        
        step = 0
        total_reward = 0
        
        while True:
            # Получаем предсказание
            action = model.predict(obs)
            
            # Выполняем шаг
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Логируем каждые 100 шагов
            if step % 100 == 0:
                logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                          f"Position={info['position_size']:.4f}, "
                          f"Total Reward={total_reward:.2f}")
            
            # Проверяем завершение
            if terminated or truncated:
                break
            
            step += 1
        
        # Финальная статистика
        logger.info("📊 Результаты миграции:")
        logger.info(f"Total steps: {step}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Final balance: {info['balance']:.2f}")
        logger.info(f"Total profit: {info['total_profit']:.4f}")
        logger.info(f"Total trades: {len(info['trades'])}")
        
        if info['trades']:
            profitable_trades = sum(1 for trade in info['trades'] if trade > 0)
            win_rate = profitable_trades / len(info['trades']) * 100
            logger.info(f"Win rate: {win_rate:.1f}%")
        
        logger.info("✅ Миграция завершена успешно")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при миграции: {e}", exc_info=True)


if __name__ == '__main__':
    run_old_model()
'''
    
    with open("run_migration.py", "w", encoding="utf-8") as f:
        f.write(migration_script)
    
    print("  ✅ Создан скрипт run_migration.py")


def create_comparison_script():
    """Создание скрипта для сравнения старой и новой версий"""
    print("📊 Создание скрипта сравнения...")
    
    comparison_script = '''#!/usr/bin/env python3
"""
Скрипт для сравнения производительности старой и новой версий
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compare_performance():
    """Сравнение производительности"""
    
    print("📊 Сравнение производительности старой и новой версий")
    print("=" * 60)
    
    # Анализ старых логов
    old_log_path = "backup_old_version/adaptive_training_trades.csv"
    if Path(old_log_path).exists():
        print("\\n📈 Анализ старой версии:")
        analyze_trades_log(old_log_path, "Старая версия")
    else:
        print("\\n⚠️  Лог старой версии не найден")
    
    # Анализ новых логов
    new_log_path = "migrated_trades.csv"
    if Path(new_log_path).exists():
        print("\\n📈 Анализ новой версии:")
        analyze_trades_log(new_log_path, "Новая версия")
    else:
        print("\\n⚠️  Лог новой версии не найден")


def analyze_trades_log(log_path, version_name):
    """Анализ лога сделок"""
    
    try:
        df = pd.read_csv(log_path)
        
        # Базовая статистика
        total_trades = len(df)
        profitable_trades = len(df[df['profit'] > 0])
        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        
        print(f"  {version_name}:")
        print(f"    Всего сделок: {total_trades}")
        print(f"    Прибыльных: {profitable_trades} ({profitable_trades/total_trades*100:.1f}%)")
        print(f"    Общая прибыль: {total_profit:.2f} USDT")
        print(f"    Средняя прибыль: {avg_profit:.2f} USDT")
        
        # Анализ по типам действий
        if 'action' in df.columns:
            action_counts = df['action'].value_counts()
            print(f"    Типы действий:")
            for action, count in action_counts.items():
                print(f"      {action}: {count} ({count/total_trades*100:.1f}%)")
        
        # Лучшие и худшие сделки
        best_trade = df['profit'].max()
        worst_trade = df['profit'].min()
        print(f"    Лучшая сделка: {best_trade:.2f} USDT")
        print(f"    Худшая сделка: {worst_trade:.2f} USDT")
        
    except Exception as e:
        print(f"    ❌ Ошибка анализа: {e}")


if __name__ == '__main__':
    compare_performance()
'''
    
    with open("compare_versions.py", "w", encoding="utf-8") as f:
        f.write(comparison_script)
    
    print("  ✅ Создан скрипт compare_versions.py")


def main():
    """Главная функция миграции"""
    print("🚀 Миграция на новую версию торгового бота")
    print("=" * 50)
    
    try:
        # Создаем резервную копию
        backup_old_files()
        
        # Мигрируем данные
        migrate_data_files()
        
        # Создаем .env файл
        create_env_file()
        
        # Обновляем конфигурацию
        update_config_for_old_data()
        
        # Создаем скрипты
        create_migration_script()
        create_comparison_script()
        
        print("\\n✅ Миграция завершена успешно!")
        print("\\n📋 Следующие шаги:")
        print("1. Добавьте ваши API ключи в файл .env")
        print("2. Установите новые зависимости: pip install -r requirements_new.txt")
        print("3. Запустите тестирование: python main.py --mode test")
        print("4. Сравните результаты: python compare_versions.py")
        print("\\n⚠️  Старые файлы сохранены в backup_old_version/")
        
    except Exception as e:
        print(f"❌ Ошибка при миграции: {e}")


if __name__ == '__main__':
    main()

