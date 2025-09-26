#!/usr/bin/env python3
"""
Тесты для валидаторов
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.validators import DataValidator, TradingValidator, validate_data_pipeline


class TestDataValidator:
    """Тесты для DataValidator"""
    
    def test_validate_ohlcv_data_valid(self):
        """Тест валидации корректных OHLCV данных"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv_data(df)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_ohlcv_data_missing_columns(self):
        """Тест валидации с отсутствующими колонками"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
            # Отсутствуют 'close' и 'volume'
        })
        
        is_valid, errors = DataValidator.validate_ohlcv_data(df)
        
        assert not is_valid
        assert 'Missing required columns' in errors[0]
    
    def test_validate_ohlcv_data_invalid_ohlc(self):
        """Тест валидации с некорректными OHLC данными"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [95, 96, 97],  # high < low - некорректно
            'low': [105, 106, 107],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv_data(df)
        
        assert not is_valid
        assert any('High price is not always >= Low price' in error for error in errors)
    
    def test_validate_ohlcv_data_negative_prices(self):
        """Тест валидации с отрицательными ценами"""
        df = pd.DataFrame({
            'open': [100, -101, 102],  # Отрицательная цена
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv_data(df)
        
        assert not is_valid
        assert any('non-positive values' in error for error in errors)
    
    def test_validate_ohlcv_data_nan_values(self):
        """Тест валидации с NaN значениями"""
        df = pd.DataFrame({
            'open': [100, 101, np.nan],  # NaN значение
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = DataValidator.validate_ohlcv_data(df)
        
        assert not is_valid
        assert any('NaN values' in error for error in errors)
    
    def test_validate_indicators_valid(self):
        """Тест валидации корректных индикаторов"""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'rsi': [50, 55, 60],
            'ema_14': [99, 100, 101],
            'macd': [0.1, 0.2, 0.3]
        })
        
        feature_list = ['close', 'volume', 'rsi', 'ema_14', 'macd']
        is_valid, errors = DataValidator.validate_indicators(df, feature_list)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_indicators_missing_features(self):
        """Тест валидации с отсутствующими признаками"""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'rsi': [50, 55, 60]
            # Отсутствуют 'ema_14' и 'macd'
        })
        
        feature_list = ['close', 'volume', 'rsi', 'ema_14', 'macd']
        is_valid, errors = DataValidator.validate_indicators(df, feature_list)
        
        assert not is_valid
        assert 'Missing features' in errors[0]
    
    def test_validate_indicators_infinite_values(self):
        """Тест валидации с бесконечными значениями"""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'rsi': [50, np.inf, 60],  # Бесконечное значение
            'ema_14': [99, 100, 101],
            'macd': [0.1, 0.2, 0.3]
        })
        
        feature_list = ['close', 'volume', 'rsi', 'ema_14', 'macd']
        is_valid, errors = DataValidator.validate_indicators(df, feature_list)
        
        assert not is_valid
        assert any('infinite values' in error for error in errors)
    
    def test_validate_trading_parameters_valid(self):
        """Тест валидации корректных торговых параметров"""
        config = {
            'sl_pct': 0.03,
            'tp_pct': 0.07,
            'fee': 0.001,
            'window_size': 24,
            'initial_deposit': 10000.0
        }
        
        is_valid, errors = DataValidator.validate_trading_parameters(config)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_trading_parameters_invalid_percentages(self):
        """Тест валидации с некорректными процентными параметрами"""
        config = {
            'sl_pct': 1.5,  # > 1.0 - некорректно
            'tp_pct': -0.01,  # < 0 - некорректно
            'fee': 0.001,
            'window_size': 24,
            'initial_deposit': 10000.0
        }
        
        is_valid, errors = DataValidator.validate_trading_parameters(config)
        
        assert not is_valid
        assert len(errors) >= 2
    
    def test_validate_model_parameters_valid(self):
        """Тест валидации корректных параметров модели"""
        config = {
            'learning_rate': 1e-4,
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 10,
            'net_arch': [512, 256, 128]
        }
        
        is_valid, errors = DataValidator.validate_model_parameters(config)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_model_parameters_invalid(self):
        """Тест валидации с некорректными параметрами модели"""
        config = {
            'learning_rate': 2.0,  # > 1.0 - некорректно
            'n_steps': -100,  # < 0 - некорректно
            'batch_size': 256,
            'n_epochs': 10,
            'net_arch': [512, -256, 128]  # Отрицательное значение
        }
        
        is_valid, errors = DataValidator.validate_model_parameters(config)
        
        assert not is_valid
        assert len(errors) >= 2


class TestTradingValidator:
    """Тесты для TradingValidator"""
    
    def test_validate_trade_action_valid(self):
        """Тест валидации корректных торговых действий"""
        assert TradingValidator.validate_trade_action(0)  # Hold
        assert TradingValidator.validate_trade_action(1)  # Buy
        assert TradingValidator.validate_trade_action(2)  # Sell
    
    def test_validate_trade_action_invalid(self):
        """Тест валидации некорректных торговых действий"""
        assert not TradingValidator.validate_trade_action(-1)
        assert not TradingValidator.validate_trade_action(3)
        assert not TradingValidator.validate_trade_action(10)
    
    def test_validate_position_size_valid(self):
        """Тест валидации корректного размера позиции"""
        is_valid, message = TradingValidator.validate_position_size(
            size=100.0,
            balance=10000.0,
            price=50000.0,
            fee=0.001
        )
        
        assert is_valid
        assert message == ""
    
    def test_validate_position_size_insufficient_balance(self):
        """Тест валидации с недостаточным балансом"""
        is_valid, message = TradingValidator.validate_position_size(
            size=1000.0,  # Слишком большой размер
            balance=1000.0,  # Недостаточный баланс
            price=50000.0,
            fee=0.001
        )
        
        assert not is_valid
        assert "Insufficient balance" in message
    
    def test_validate_position_size_negative(self):
        """Тест валидации с отрицательным размером позиции"""
        is_valid, message = TradingValidator.validate_position_size(
            size=-100.0,  # Отрицательный размер
            balance=10000.0,
            price=50000.0,
            fee=0.001
        )
        
        assert not is_valid
        assert "must be positive" in message
    
    def test_validate_price_valid(self):
        """Тест валидации корректной цены"""
        assert TradingValidator.validate_price(50000.0)
        assert TradingValidator.validate_price(100.0)
        assert TradingValidator.validate_price(0.001)
    
    def test_validate_price_invalid(self):
        """Тест валидации некорректной цены"""
        assert not TradingValidator.validate_price(-100.0)  # Отрицательная
        assert not TradingValidator.validate_price(0.0)  # Нулевая
        assert not TradingValidator.validate_price(np.nan)  # NaN
        assert not TradingValidator.validate_price(np.inf)  # Бесконечная


class TestValidateDataPipeline:
    """Тесты для validate_data_pipeline"""
    
    def test_validate_data_pipeline_valid(self):
        """Тест валидации корректного пайплайна данных"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200],
            'rsi': [50, 55, 60],
            'ema_14': [99, 100, 101],
            'macd': [0.1, 0.2, 0.3]
        })
        
        feature_list = ['close', 'volume', 'rsi', 'ema_14', 'macd']
        
        is_valid = validate_data_pipeline(df, feature_list)
        
        assert is_valid
    
    def test_validate_data_pipeline_invalid_ohlcv(self):
        """Тест валидации с некорректными OHLCV данными"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [95, 96, 97],  # high < low - некорректно
            'low': [105, 106, 107],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200],
            'rsi': [50, 55, 60],
            'ema_14': [99, 100, 101],
            'macd': [0.1, 0.2, 0.3]
        })
        
        feature_list = ['close', 'volume', 'rsi', 'ema_14', 'macd']
        
        is_valid = validate_data_pipeline(df, feature_list)
        
        assert not is_valid
