#!/usr/bin/env python3
"""
Главный файл для запуска торгового бота с улучшенной логикой
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import get_config, get_api_credentials
from src.utils.logging import setup_logging, get_logger
from src.utils.monitoring import TradingMonitor
from src.data.data_loader import DataLoader
from src.env.improved_trading_env import ImprovedTradingEnv
from src.models.ppo_model import PPOModel
from api import get_bybit_client
from trade import market_buy, market_sell


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Improved Crypto Trading Bot')
    parser.add_argument('--config', type=str, default='config/improved_trading_config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'trade', 'test'], 
                       default='test', help='Mode: train, trade, or test')
    parser.add_argument('--model-path', type=str, default='improved_ppo_model.zip',
                       help='Path to model file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with old logic')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        config = get_config(args.config)
        
        # Настраиваем логирование
        logger = setup_logging(config.logging, "main_improved")
        logger.info("🚀 Starting Improved Crypto Trading Bot")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Config: {args.config}")
        
        # Инициализируем мониторинг
        monitor = TradingMonitor(config.monitoring)
        
        # Сбрасываем метрики для чистого старта
        monitor.reset_metrics()
        
        if args.mode == 'train':
            train_mode(config, args, logger, monitor)
        elif args.mode == 'trade':
            trade_mode(config, args, logger, monitor)
        elif args.mode == 'test':
            test_mode(config, args, logger, monitor)
        
        if args.compare:
            compare_logic(logger)
        
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'logger' in locals():
            logger.error(f"Fatal error: {e}", exc_info=True)


def train_mode(config, args, logger, monitor):
    """Режим обучения с улучшенной логикой"""
    logger.info("🤖 Starting training mode with improved logic")
    
    try:
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data(args.data_path)
        
        # Создаем обучающие данные
        df_train = data_loader.create_training_data(df)
        
        # Проверяем на NaN и бесконечные значения
        logger.info("Checking data for NaN and infinite values...")
        nan_count = df_train.isna().sum().sum()
        inf_count = np.isinf(df_train.select_dtypes(include=[np.number])).sum().sum()
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in training data, filling with 0")
            df_train = df_train.fillna(0)
        
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in training data, replacing with large finite values")
            df_train = df_train.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Проверяем, что все значения конечны
        if not np.isfinite(df_train.select_dtypes(include=[np.number])).all().all():
            logger.error("Data still contains non-finite values after cleaning")
            raise ValueError("Training data contains non-finite values")
        
        logger.info("Data validation passed - all values are finite")
        
        # Сохраняем временные данные ПЕРЕД созданием среды
        df_train.to_csv('temp_improved_training_data.csv', index=False)
        
        # Создаем улучшенную среду
        env = ImprovedTradingEnv(
            df_path='temp_improved_training_data.csv',
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='improved_training_trades.csv'
        )
        
        # Создаем модель
        model_config = config.model.__dict__.copy()
        model_path = args.model_path or 'improved_ppo_model.zip'
        model_config['model_path'] = model_path
        model = PPOModel(model_config)

        # Проверяем, существует ли модель для продолжения обучения
        resume_training = False
        existing_model_path = None

        if os.path.exists(model_path):
            existing_model_path = model_path
        elif os.path.exists('ppo_adaptive_model.zip'):
            existing_model_path = 'ppo_adaptive_model.zip'

        if existing_model_path:
            try:
                logger.info(f"📂 Found existing model at {existing_model_path}, attempting to resume training...")
                resume_training = True
                model_path = existing_model_path
                model_config['model_path'] = model_path
            except Exception as e:
                logger.warning(f"❌ Error with existing model: {e}, starting fresh training")
                resume_training = False
                existing_model_path = None

        # Загружаем модель для продолжения обучения если нужно
        if resume_training and existing_model_path:
            try:
                logger.info("Loading model for resume training...")
                # Загружаем модель вместе со средой для продолжения обучения
                model.load(existing_model_path, env=env)
                current_timesteps = getattr(model.model, 'num_timesteps', 0)
                remaining_timesteps = max(0, config.model.total_timesteps - current_timesteps)
                logger.info(f"📊 Current timesteps: {current_timesteps}, remaining: {remaining_timesteps}")

                if remaining_timesteps <= 0:
                    logger.info("🎯 Training already completed, skipping...")
                    return

                total_timesteps = remaining_timesteps
                logger.info("✅ Model loaded successfully for resume training")

            except Exception as e:
                logger.warning(f"❌ Failed to load existing model: {e}")
                logger.info("🔄 Starting fresh training instead...")
                resume_training = False
                total_timesteps = config.model.total_timesteps
        else:
            total_timesteps = config.model.total_timesteps

        # Обучаем модель
        logger.info(f"🎯 {'Resuming' if resume_training else 'Starting'} training for {total_timesteps} timesteps")
        logger.info("📊 Improved logic features:")
        logger.info(f"   - Min profit threshold: {env.improved_config.min_profit_threshold} USDT")
        logger.info(f"   - Trailing stop multiplier: {env.improved_config.trailing_stop_multiplier}")
        logger.info(f"   - Trailing TP arm threshold: {env.improved_config.trailing_tp_arm_threshold}")
        logger.info(f"   - Dynamic SL enabled: {env.improved_config.dynamic_sl_enabled}")
        
        result = model.train(env, total_timesteps=total_timesteps, reset_num_timesteps=not resume_training)
        
        if result['success']:
            logger.info("✅ Training completed successfully")
            logger.info(f"Model saved to: {result['model_path']}")
        else:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            logger.error(f"Full result: {result}")
        
        # Очищаем временные файлы
        if os.path.exists('temp_improved_training_data.csv'):
            os.remove('temp_improved_training_data.csv')
        
    except Exception as e:
        logger.error(f"Error in training mode: {e}", exc_info=True)
        raise


def trade_mode(config, args, logger, monitor):
    """Режим торговли с улучшенной логикой"""
    logger.info("💰 Starting trading mode with improved logic")
    
    try:
        # Получаем API ключи
        api_key, api_secret = get_api_credentials(config)
        client = get_bybit_client(testnet=config.api.testnet)
        
        # Загружаем модель
        model_config = config.model.__dict__.copy()
        model_config['model_path'] = args.model_path
        model = PPOModel(model_config)
        
        if os.path.exists(args.model_path):
            model.load(args.model_path)
            logger.info(f"✅ Model loaded from {args.model_path}")
        else:
            logger.error(f"❌ Model file not found: {args.model_path}")
            return
        
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.get_ohlcv_from_bybit(
            client, 
            config.trading.symbol, 
            config.data.interval, 
            config.data.limit
        )
        
        if df is None:
            logger.error("❌ Failed to load data from Bybit")
            return
        
        # Создаем улучшенную среду
        env = ImprovedTradingEnv(
            df_path='temp_improved_live_data.csv',
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='improved_live_trades.csv'
        )
        
        # Сохраняем данные
        df.to_csv('temp_improved_live_data.csv', index=False)
        
        # Сбрасываем среду
        obs, info = env.reset()
        
        logger.info("🚀 Starting live trading with improved logic")
        logger.info(f"Initial balance: {info['balance']:.2f} USDT")
        logger.info("📊 Improved features active:")
        logger.info(f"   - Min profit threshold: {env.improved_config.min_profit_threshold} USDT")
        logger.info(f"   - Trailing TP arm threshold: {env.improved_config.trailing_tp_arm_threshold}")
        logger.info(f"   - Dynamic stop loss: {env.improved_config.dynamic_sl_enabled}")
        
        step = 0
        while True:
            try:
                # Получаем предсказание от модели
                start_time = time.time()
                action = model.predict(obs)
                prediction_time = time.time() - start_time
                
                # Выполняем шаг в среде
                step_start_time = time.time()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time = time.time() - step_start_time
                
                # Обновляем метрики
                monitor.update_trading_metrics(step, info)
                monitor.update_performance_metrics(step, prediction_time, step_time)
                
                # Логируем состояние
                if step % 100 == 0:
                    position_info = info.get('position_info', {})
                    logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                              f"Position={position_info.get('position_size', 0):.4f}, "
                              f"Profit={info['total_profit']:.4f}, "
                              f"Volatility={info.get('volatility', 0):.3f}")
                
                # Проверяем завершение
                if terminated or truncated:
                    logger.info("Episode finished, resetting environment")
                    obs, info = env.reset()
                
                step += 1
                
                # Пауза между итерациями
                time.sleep(60)  # 1 минута
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Пауза при ошибке
        
        # Финальная статистика
        summary = monitor.get_summary()
        logger.info("📊 Final Statistics:")
        logger.info(f"Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"Total trades: {summary.get('total_trades', 0)}")
        logger.info(f"Win rate: {summary.get('win_rate', 0):.1f}%")
        logger.info(f"Total profit: {summary.get('total_profit', 0):.2f} USDT")
        logger.info(f"Final balance: {summary.get('current_balance', 0):.2f} USDT")
        
        # Экспортируем метрики
        monitor.export_to_csv()
        
        # Очищаем временные файлы
        if os.path.exists('temp_improved_live_data.csv'):
            os.remove('temp_improved_live_data.csv')
        
    except Exception as e:
        logger.error(f"Error in trading mode: {e}", exc_info=True)
        raise


def test_mode(config, args, logger, monitor):
    """Режим тестирования с улучшенной логикой"""
    logger.info("🧪 Starting test mode with improved logic")
    
    try:
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data(args.data_path)
        
        # Создаем улучшенную среду
        env = ImprovedTradingEnv(
            df_path=args.data_path or config.data.dataset_path,
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='improved_test_trades.csv'
        )
        
        # Загружаем модель если есть
        model = None
        if os.path.exists(args.model_path):
            model_config = config.model.__dict__.copy()
            model_config['model_path'] = args.model_path
            model = PPOModel(model_config)
            model.load(args.model_path)
            logger.info(f"✅ Model loaded from {args.model_path}")
        else:
            logger.warning(f"⚠️  Model file not found: {args.model_path}, using random actions")
        
        # Сбрасываем среду
        obs, info = env.reset()
        
        logger.info("🧪 Starting backtest with improved logic")
        logger.info(f"Initial balance: {info['balance']:.2f} USDT")
        logger.info("📊 Improved features:")
        logger.info(f"   - Min profit threshold: {env.improved_config.min_profit_threshold} USDT")
        logger.info(f"   - Trailing stop multiplier: {env.improved_config.trailing_stop_multiplier}")
        logger.info(f"   - Trailing TP arm threshold: {env.improved_config.trailing_tp_arm_threshold}")
        logger.info(f"   - Partial TP ratio: {env.improved_config.partial_close_ratio}")
        
        step = 0
        while True:
            try:
                # Получаем действие
                if model is not None:
                    action = model.predict(obs)
                else:
                    action = env.action_space.sample()  # Случайное действие
                
                # Выполняем шаг
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Обновляем метрики
                monitor.update_trading_metrics(step, info)
                
                # Логируем каждые 100 шагов
                if step % 100 == 0:
                    position_info = info.get('position_info', {})
                    logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                              f"Position={position_info.get('position_size', 0):.4f}, "
                              f"Profit={info['total_profit']:.4f}, "
                              f"Volatility={info.get('volatility', 0):.3f}")
                
                # Проверяем завершение
                if terminated or truncated:
                    break
                
                step += 1
                
            except Exception as e:
                logger.error(f"Error in test loop: {e}")
                break
        
        # Финальная статистика
        summary = monitor.get_summary()
        logger.info("📊 Test Results:")
        logger.info(f"Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"Total trades: {summary.get('total_trades', 0)}")
        logger.info(f"Win rate: {summary.get('win_rate', 0):.1f}%")
        logger.info(f"Total profit: {summary.get('total_profit', 0):.2f} USDT")
        logger.info(f"Final balance: {summary.get('current_balance', 0):.2f} USDT")
        logger.info(f"Max drawdown: {summary.get('max_drawdown', 0):.2%}")
        logger.info(f"Sharpe ratio: {summary.get('sharpe_ratio', 0):.3f}")
        
        # Статистика позиции
        position_info = info.get('position_info', {})
        if position_info:
            logger.info("📈 Position Statistics:")
            logger.info(f"  Total trades: {position_info.get('total_trades', 0)}")
            logger.info(f"  Profitable trades: {position_info.get('profitable_trades', 0)}")
            logger.info(f"  Win rate: {position_info.get('win_rate', 0):.1f}%")
            logger.info(f"  Total profit: {position_info.get('total_profit', 0):.2f} USDT")
        
        # Экспортируем метрики
        monitor.export_to_csv()
        
    except Exception as e:
        logger.error(f"Error in test mode: {e}", exc_info=True)
        raise


def compare_logic(logger):
    """Сравнение старой и новой логики"""
    logger.info("📊 Comparing old vs improved logic")
    
    try:
        import pandas as pd
        
        # Анализируем старые логи
        if os.path.exists('adaptive_training_trades.csv'):
            old_df = pd.read_csv('adaptive_training_trades.csv')
            
            logger.info("📈 Old Logic Results:")
            logger.info(f"  - Total trades: {len(old_df)}")
            logger.info(f"  - Average profit: {old_df['profit'].mean():.2f} USDT")
            logger.info(f"  - Micro trades (< 10 USDT): {len(old_df[abs(old_df['profit']) < 10])} ({len(old_df[abs(old_df['profit']) < 10])/len(old_df)*100:.1f}%)")
            logger.info(f"  - Stop losses: {len(old_df[old_df['action'] == 'sell_stop_loss'])} ({len(old_df[old_df['action'] == 'sell_stop_loss'])/len(old_df)*100:.1f}%)")
            logger.info(f"  - Trailing TP: {len(old_df[old_df['action'] == 'sell_trailing_tp'])} ({len(old_df[old_df['action'] == 'sell_trailing_tp'])/len(old_df)*100:.1f}%)")
            logger.info(f"  - Partial TP: {len(old_df[old_df['action'] == 'partial_tp'])} ({len(old_df[old_df['action'] == 'partial_tp'])/len(old_df)*100:.1f}%)")
        
        # Анализируем новые логи
        if os.path.exists('improved_test_trades.csv'):
            new_df = pd.read_csv('improved_test_trades.csv')
            
            logger.info("📈 Improved Logic Results:")
            logger.info(f"  - Total trades: {len(new_df)}")
            logger.info(f"  - Average profit: {new_df['profit'].mean():.2f} USDT")
            logger.info(f"  - Micro trades (< 15 USDT): {len(new_df[abs(new_df['profit']) < 15])} ({len(new_df[abs(new_df['profit']) < 15])/len(new_df)*100:.1f}%)")
            logger.info(f"  - Stop losses: {len(new_df[new_df['action'] == 'sell_stop_loss'])} ({len(new_df[new_df['action'] == 'sell_stop_loss'])/len(new_df)*100:.1f}%)")
            logger.info(f"  - Trailing TP: {len(new_df[new_df['action'] == 'sell_trailing_tp'])} ({len(new_df[new_df['action'] == 'sell_trailing_tp'])/len(new_df)*100:.1f}%)")
            logger.info(f"  - Partial TP: {len(new_df[new_df['action'] == 'partial_tp'])} ({len(new_df[new_df['action'] == 'partial_tp'])/len(new_df)*100:.1f}%)")
        
        logger.info("💡 Expected Improvements:")
        logger.info("  ✅ Reduced micro trades by 30-50%")
        logger.info("  ✅ Fewer frequent stop losses by 20-30%")
        logger.info("  ✅ Activated partial take profits")
        logger.info("  ✅ More stable profitability")
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")


if __name__ == '__main__':
    main()

