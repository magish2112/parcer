#!/usr/bin/env python3
"""
Тестирование улучшенной торговой логики
"""

import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import get_config
from src.utils.logging import setup_logging
from src.data.data_loader import DataLoader
from src.env.improved_trading_env import ImprovedTradingEnv
from src.models.ppo_model import PPOModel


def test_improved_logic():
    """Тестирование улучшенной логики"""
    
    print("🧪 Тестирование улучшенной торговой логики")
    print("=" * 50)
    
    try:
        # Загружаем улучшенную конфигурацию
        config = get_config('config/improved_trading_config.yaml')
        
        # Настраиваем логирование
        logger = setup_logging(config.logging, "test_improved")
        logger.info("Starting improved logic test")
        
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data()
        
        # Сохраняем данные
        df.to_csv('temp_test_data.csv', index=False)
        
        # Создаем улучшенную среду
        env = ImprovedTradingEnv(
            df_path='temp_test_data.csv',
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='improved_test_trades.csv'
        )
        
        # Сбрасываем среду
        obs, info = env.reset()
        
        logger.info(f"🚀 Начальное состояние: Balance={info['balance']:.2f}")
        logger.info(f"📊 Параметры улучшенной логики:")
        logger.info(f"   - Min profit threshold: {env.improved_config.min_profit_threshold} USDT")
        logger.info(f"   - Trailing stop multiplier: {env.improved_config.trailing_stop_multiplier}")
        logger.info(f"   - Trailing TP arm threshold: {env.improved_config.trailing_tp_arm_threshold}")
        logger.info(f"   - Partial TP ratio: {env.improved_config.partial_close_ratio}")
        
        step = 0
        total_reward = 0
        action_count = {0: 0, 1: 0, 2: 0}  # hold, buy, sell
        
        while True:
            # Получаем случайное действие для тестирования
            action = env.action_space.sample()
            action_count[action] += 1
            
            # Выполняем шаг
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Логируем каждые 100 шагов
            if step % 100 == 0:
                position_info = info.get('position_info', {})
                logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                          f"Position={position_info.get('position_size', 0):.4f}, "
                          f"Total Reward={total_reward:.2f}, "
                          f"Volatility={info.get('volatility', 0):.3f}")
                
                # Логируем состояние позиции
                if position_info.get('position_size', 0) > 0:
                    logger.info(f"  Position: Entry={position_info.get('avg_entry_price', 0):.2f}, "
                              f"Trailing TP Active={position_info.get('trailing_tp_active', False)}, "
                              f"Trailing TP Armed={position_info.get('trailing_tp_armed', False)}, "
                              f"Partial Closes={position_info.get('partial_closes_count', 0)}")
            
            # Проверяем завершение
            if terminated or truncated:
                break
            
            step += 1
        
        # Финальная статистика
        logger.info("📊 Результаты тестирования улучшенной логики:")
        logger.info(f"Total steps: {step}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Final balance: {info['balance']:.2f}")
        logger.info(f"Total profit: {info['total_profit']:.4f}")
        logger.info(f"Total trades: {len(info['trades'])}")
        
        # Статистика действий
        total_actions = sum(action_count.values())
        logger.info(f"Action distribution:")
        for action, count in action_count.items():
            action_name = ['Hold', 'Buy', 'Sell'][action]
            logger.info(f"  {action_name}: {count} ({count/total_actions*100:.1f}%)")
        
        # Статистика позиции
        position_info = info.get('position_info', {})
        if position_info:
            logger.info(f"Position statistics:")
            logger.info(f"  Total trades: {position_info.get('total_trades', 0)}")
            logger.info(f"  Profitable trades: {position_info.get('profitable_trades', 0)}")
            logger.info(f"  Win rate: {position_info.get('win_rate', 0):.1f}%")
            logger.info(f"  Total profit: {position_info.get('total_profit', 0):.2f} USDT")
        
        # Анализ сделок
        if info['trades']:
            trades = info['trades']
            profitable_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            
            logger.info(f"Trade analysis:")
            logger.info(f"  Profitable trades: {len(profitable_trades)} ({len(profitable_trades)/len(trades)*100:.1f}%)")
            logger.info(f"  Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
            
            if profitable_trades:
                logger.info(f"  Avg profitable trade: {sum(profitable_trades)/len(profitable_trades):.2f} USDT")
            if losing_trades:
                logger.info(f"  Avg losing trade: {sum(losing_trades)/len(losing_trades):.2f} USDT")
            
            # Анализ микросделок
            micro_trades = [t for t in trades if abs(t) < env.improved_config.min_profit_threshold]
            logger.info(f"  Micro trades (< {env.improved_config.min_profit_threshold} USDT): {len(micro_trades)} ({len(micro_trades)/len(trades)*100:.1f}%)")
        
        logger.info("✅ Тестирование улучшенной логики завершено")
        
        # Очищаем временные файлы
        if os.path.exists('temp_test_data.csv'):
            os.remove('temp_test_data.csv')
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}", exc_info=True)


def compare_with_old_logic():
    """Сравнение с старой логикой"""
    
    print("\n📊 Сравнение с старой логикой")
    print("=" * 50)
    
    try:
        import pandas as pd
        
        # Анализируем старые логи
        if os.path.exists('adaptive_training_trades.csv'):
            old_df = pd.read_csv('adaptive_training_trades.csv')
            
            print("📈 Старая логика:")
            print(f"  - Всего сделок: {len(old_df)}")
            print(f"  - Средняя прибыль: {old_df['profit'].mean():.2f} USDT")
            print(f"  - Микросделки (< 10 USDT): {len(old_df[abs(old_df['profit']) < 10])} ({len(old_df[abs(old_df['profit']) < 10])/len(old_df)*100:.1f}%)")
            print(f"  - Стоп-лоссы: {len(old_df[old_df['action'] == 'sell_stop_loss'])} ({len(old_df[old_df['action'] == 'sell_stop_loss'])/len(old_df)*100:.1f}%)")
            print(f"  - Трейлинг-ТП: {len(old_df[old_df['action'] == 'sell_trailing_tp'])} ({len(old_df[old_df['action'] == 'sell_trailing_tp'])/len(old_df)*100:.1f}%)")
        
        # Анализируем новые логи
        if os.path.exists('improved_test_trades.csv'):
            new_df = pd.read_csv('improved_test_trades.csv')
            
            print("\n📈 Улучшенная логика:")
            print(f"  - Всего сделок: {len(new_df)}")
            print(f"  - Средняя прибыль: {new_df['profit'].mean():.2f} USDT")
            print(f"  - Микросделки (< 15 USDT): {len(new_df[abs(new_df['profit']) < 15])} ({len(new_df[abs(new_df['profit']) < 15])/len(new_df)*100:.1f}%)")
            print(f"  - Стоп-лоссы: {len(new_df[new_df['action'] == 'sell_stop_loss'])} ({len(new_df[new_df['action'] == 'sell_stop_loss'])/len(new_df)*100:.1f}%)")
            print(f"  - Трейлинг-ТП: {len(new_df[new_df['action'] == 'sell_trailing_tp'])} ({len(new_df[new_df['action'] == 'sell_trailing_tp'])/len(new_df)*100:.1f}%)")
            print(f"  - Частичные ТП: {len(new_df[new_df['action'] == 'partial_tp'])} ({len(new_df[new_df['action'] == 'partial_tp'])/len(new_df)*100:.1f}%)")
        
        print("\n💡 Ожидаемые улучшения:")
        print("  ✅ Сокращение микросделок на 30-50%")
        print("  ✅ Уменьшение частых стоп-лоссов на 20-30%")
        print("  ✅ Активация частичных тейк-профитов")
        print("  ✅ Более стабильная прибыльность")
        
    except Exception as e:
        print(f"❌ Ошибка при сравнении: {e}")


if __name__ == '__main__':
    test_improved_logic()
    compare_with_old_logic()
