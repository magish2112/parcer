#!/usr/bin/env python3
"""
Главный файл для запуска торгового бота с новой архитектурой
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import get_config, get_api_credentials
from src.utils.logging import setup_logging, get_logger
from src.utils.monitoring import TradingMonitor
from src.data.data_loader import DataLoader
from src.env.trading_env import OptimizedTradingEnv
from src.models.ppo_model import PPOModel
from api import get_bybit_client
from trade import market_buy, market_sell


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--config', type=str, default='config/trading_config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'trade', 'test'], 
                       default='trade', help='Mode: train, trade, or test')
    parser.add_argument('--model-path', type=str, default='ppo_model.zip',
                       help='Path to model file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию
        config = get_config(args.config)
        
        # Настраиваем логирование
        logger = setup_logging(config.logging, "main")
        logger.info("Starting Crypto Trading Bot")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Config: {args.config}")
        
        # Инициализируем мониторинг
        monitor = TradingMonitor(config.monitoring)
        
        if args.mode == 'train':
            train_mode(config, args, logger, monitor)
        elif args.mode == 'trade':
            trade_mode(config, args, logger, monitor)
        elif args.mode == 'test':
            test_mode(config, args, logger, monitor)
        
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'logger' in locals():
            logger.error(f"Fatal error: {e}", exc_info=True)


def train_mode(config, args, logger, monitor):
    """Режим обучения"""
    logger.info("🤖 Starting training mode")
    
    try:
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data(args.data_path)
        
        # Создаем обучающие данные
        df_train = data_loader.create_training_data(df)
        
        # Создаем среду
        env = OptimizedTradingEnv(
            df_path='temp_training_data.csv',
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='training_trades.csv'
        )
        
        # Сохраняем временные данные
        df_train.to_csv('temp_training_data.csv', index=False)
        
        # Создаем модель
        model_config = config.model.__dict__.copy()
        model_config['model_path'] = args.model_path
        model = PPOModel(model_config)
        
        # Обучаем модель
        logger.info(f"🎯 Starting training for {config.model.total_timesteps} timesteps")
        
        result = model.train(env, total_timesteps=config.model.total_timesteps)
        
        if result['success']:
            logger.info("✅ Training completed successfully")
            logger.info(f"Model saved to: {result['model_path']}")
        else:
            logger.error(f"❌ Training failed: {result['error']}")
        
        # Очищаем временные файлы
        if os.path.exists('temp_training_data.csv'):
            os.remove('temp_training_data.csv')
        
    except Exception as e:
        logger.error(f"Error in training mode: {e}", exc_info=True)
        raise


def trade_mode(config, args, logger, monitor):
    """Режим торговли"""
    logger.info("💰 Starting trading mode")
    
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
        
        # Создаем среду
        env = OptimizedTradingEnv(
            df_path='temp_live_data.csv',
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='live_trades.csv'
        )
        
        # Сохраняем данные
        df.to_csv('temp_live_data.csv', index=False)
        
        # Сбрасываем среду
        obs, info = env.reset()
        
        logger.info("🚀 Starting live trading")
        logger.info(f"Initial balance: {info['balance']:.2f} USDT")
        
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
                    logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                              f"Position={info['position_size']:.4f}, "
                              f"Profit={info['total_profit']:.4f}")
                
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
        if os.path.exists('temp_live_data.csv'):
            os.remove('temp_live_data.csv')
        
    except Exception as e:
        logger.error(f"Error in trading mode: {e}", exc_info=True)
        raise


def test_mode(config, args, logger, monitor):
    """Режим тестирования"""
    logger.info("🧪 Starting test mode")
    
    try:
        # Загружаем данные
        data_loader = DataLoader(config.data.__dict__)
        df = data_loader.load_historical_data(args.data_path)
        
        # Создаем среду
        env = OptimizedTradingEnv(
            df_path=args.data_path or config.data.dataset_path,
            trading_config=config.trading,
            data_config=config.data,
            logging_config=config.logging,
            log_trades_path='test_trades.csv'
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
        
        logger.info("🧪 Starting backtest")
        logger.info(f"Initial balance: {info['balance']:.2f} USDT")
        
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
                    logger.info(f"Step {step}: Balance={info['balance']:.2f}, "
                              f"Position={info['position_size']:.4f}, "
                              f"Profit={info['total_profit']:.4f}")
                
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
        
        # Экспортируем метрики
        monitor.export_to_csv()
        
    except Exception as e:
        logger.error(f"Error in test mode: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
