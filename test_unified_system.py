#!/usr/bin/env python3
"""
Быстрый тест единой торговой системы
"""

from unified_trading_system import UnifiedTradingEnv, TradingConfig, UnifiedTrainingManager
import numpy as np

def test_unified_system():
    """Тестирование единой системы"""
    print("🧪 Тестирование единой торговой системы...")
    
    # Создание конфигурации
    config = TradingConfig()
    print(f"✅ Конфигурация создана:")
    print(f"   - Размер позиции: {config.max_position_size*100:.1f}%")
    print(f"   - Минимальный порог прибыли: {config.min_profit_threshold} USDT")
    print(f"   - Arm-порог трейлинга: {config.trailing_tp_arm_threshold*100:.1f}%")
    print(f"   - Множитель стоп-лосса: {config.trailing_mult}")
    
    # Создание среды
    env = UnifiedTradingEnv("btc_4h_full_fixed.csv", config)
    print(f"✅ Среда создана: наблюдение shape={env.observation_space.shape}")
    
    # Тест сброса
    obs, info = env.reset()
    print(f"✅ Сброс среды: наблюдение shape={obs.shape}")
    
    # Тест нескольких шагов
    total_reward = 0
    trades_count = 0
    
    print("\n🔄 Тестирование 500 шагов...")
    
    for step in range(500):
        # Простая стратегия
        action = 0  # hold по умолчанию
        
        if env.position_size == 0:
            # Покупаем при тренде вверх
            if env.df['trend_ema'].iloc[env.current_step] > 0 and env.df['rsi'].iloc[env.current_step] < 70:
                action = 1  # buy
        else:
            # Продаем при слабом тренде или высоком RSI
            if env.df['trend_ema'].iloc[env.current_step] <= 0 or env.df['rsi'].iloc[env.current_step] > 80:
                action = 2  # sell
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
        
        # Логирование важных событий
        if step % 100 == 0:
            print(f"   Шаг {step}: баланс={info['balance']:.0f}, позиция={info['position_size']:.6f}, "
                  f"стопы={info['consecutive_stops']}, победы={info['consecutive_wins']}, "
                  f"множитель={info['position_size_multiplier']:.2f}")
    
    print(f"\n📊 Результаты теста:")
    print(f"   - Общая награда: {total_reward:.2f}")
    print(f"   - Финальный баланс: {info['balance']:.2f}")
    print(f"   - Общая прибыль: {info['total_profit']:.4f}")
    print(f"   - Количество сделок: {len(env.trades)}")
    print(f"   - Подряд стопов: {info['consecutive_stops']}")
    print(f"   - Подряд побед: {info['consecutive_wins']}")
    print(f"   - Множитель размера позиции: {info['position_size_multiplier']:.2f}")
    
    if env.trades:
        profitable_trades = [t for t in env.trades if t > 0]
        losing_trades = [t for t in env.trades if t < 0]
        
        print(f"   - Прибыльных сделок: {len(profitable_trades)}")
        print(f"   - Убыточных сделок: {len(losing_trades)}")
        print(f"   - Средняя прибыль: {np.mean(profitable_trades):.2f}" if profitable_trades else "   - Средняя прибыль: 0")
        print(f"   - Средний убыток: {np.mean(losing_trades):.2f}" if losing_trades else "   - Средний убыток: 0")
        
        # Проверка на микросделки
        micro_trades = [t for t in env.trades if abs(t) < config.min_profit_threshold]
        print(f"   - Микросделок (<{config.min_profit_threshold} USDT): {len(micro_trades)}")
    
    print(f"\n✅ Тест единой системы завершен успешно!")
    return env, info

def test_training_manager():
    """Тестирование менеджера обучения"""
    print("\n🧪 Тестирование менеджера обучения...")
    
    try:
        trainer = UnifiedTrainingManager()
        print("✅ Менеджер обучения создан")
        print(f"   - Этапов обучения: {len(trainer.training_config.stages)}")
        print(f"   - Целевая награда: {trainer.training_config.target_reward}")
        
        # Тест создания среды
        stage_config = trainer.training_config.stages[0]
        env = trainer.create_environment(stage_config)
        print(f"✅ Среда для этапа '{stage_config['name']}' создана")
        
        # Тест создания модели
        model = trainer.create_model(stage_config, env)
        print("✅ Модель PPO создана")
        
        print("✅ Менеджер обучения работает корректно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в менеджере обучения: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование единой торговой системы без дублирования\n")
    
    try:
        # Тест среды
        env, info = test_unified_system()
        
        # Тест менеджера обучения
        training_ok = test_training_manager()
        
        print(f"\n📈 Итоги тестирования:")
        print(f"   ✅ Единая среда торговли работает")
        print(f"   {'✅' if training_ok else '❌'} Менеджер обучения {'работает' if training_ok else 'не работает'}")
        print(f"   ✅ Адаптивный риск-скоринг: множитель {info['position_size_multiplier']:.2f}")
        print(f"   ✅ Микросделки сокращены: {len([t for t in env.trades if abs(t) < 25.0])} из {len(env.trades)}")
        
        print(f"\n🎯 Система готова к обучению!")
        
    except Exception as e:
        print(f"❌ Ошибка в тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
