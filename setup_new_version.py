#!/usr/bin/env python3
"""
Скрипт для настройки новой версии торгового бота
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Проверка версии Python"""
    print("🐍 Проверка версии Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Установка зависимостей"""
    print("📦 Установка зависимостей...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_new.txt"])
        print("✅ Зависимости установлены успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        return False


def create_directories():
    """Создание необходимых директорий"""
    print("📁 Создание директорий...")
    
    directories = [
        "logs",
        "logs/metrics",
        "data",
        "models",
        "tensorboard_logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    return True


def check_config():
    """Проверка конфигурации"""
    print("⚙️  Проверка конфигурации...")
    
    config_path = "config/trading_config.yaml"
    if not Path(config_path).exists():
        print(f"❌ Файл конфигурации не найден: {config_path}")
        return False
    
    print(f"✅ Конфигурация найдена: {config_path}")
    return True


def check_env_file():
    """Проверка файла .env"""
    print("🔐 Проверка файла .env...")
    
    env_path = ".env"
    if not Path(env_path).exists():
        print("⚠️  Файл .env не найден")
        print("📝 Создайте файл .env с вашими API ключами:")
        print("   BYBIT_API_KEY=your_api_key_here")
        print("   BYBIT_API_SECRET=your_api_secret_here")
        return False
    
    print("✅ Файл .env найден")
    return True


def run_tests():
    """Запуск тестов"""
    print("🧪 Запуск тестов...")
    
    try:
        # Проверяем наличие pytest
        subprocess.check_call([sys.executable, "-m", "pytest", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Запускаем тесты
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Все тесты прошли успешно")
            return True
        else:
            print("⚠️  Некоторые тесты не прошли:")
            print(result.stdout)
            return False
            
    except subprocess.CalledProcessError:
        print("⚠️  pytest не установлен, пропускаем тесты")
        return True
    except Exception as e:
        print(f"⚠️  Ошибка запуска тестов: {e}")
        return True


def create_sample_data():
    """Создание примеров данных"""
    print("📊 Создание примеров данных...")
    
    # Проверяем наличие данных
    data_files = [
        "btc_4h_full_fixed.csv",
        "data/btc_4h_full_fixed.csv"
    ]
    
    data_found = False
    for file in data_files:
        if Path(file).exists():
            print(f"✅ Данные найдены: {file}")
            data_found = True
            break
    
    if not data_found:
        print("⚠️  Файлы данных не найдены")
        print("📝 Поместите файл btc_4h_full_fixed.csv в корневую директорию или data/")
    
    return data_found


def main():
    """Главная функция настройки"""
    print("🚀 Настройка новой версии торгового бота")
    print("=" * 50)
    
    success = True
    
    # Проверяем версию Python
    if not check_python_version():
        success = False
    
    # Создаем директории
    if not create_directories():
        success = False
    
    # Проверяем конфигурацию
    if not check_config():
        success = False
    
    # Проверяем .env файл
    if not check_env_file():
        success = False
    
    # Устанавливаем зависимости
    if not install_dependencies():
        success = False
    
    # Запускаем тесты
    if not run_tests():
        success = False
    
    # Проверяем данные
    if not create_sample_data():
        success = False
    
    print("\\n" + "=" * 50)
    
    if success:
        print("✅ Настройка завершена успешно!")
        print("\\n🚀 Готово к использованию:")
        print("   python main.py --mode test    # Тестирование")
        print("   python main.py --mode train   # Обучение")
        print("   python main.py --mode trade   # Торговля")
    else:
        print("⚠️  Настройка завершена с предупреждениями")
        print("\\n📋 Проверьте:")
        print("   - Файл .env с API ключами")
        print("   - Файлы данных")
        print("   - Конфигурацию")
    
    print("\\n📚 Документация: README_NEW.md")
    print("🔄 Миграция: python migrate_to_new_version.py")


if __name__ == '__main__':
    main()

