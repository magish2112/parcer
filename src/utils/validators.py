#!/usr/bin/env python3
"""
Валидация данных для торгового бота
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .logging import get_logger

logger = get_logger("validators")


class DataValidator:
    """Валидатор данных"""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Валидация OHLCV данных"""
        errors = []
        
        # Проверяем наличие обязательных колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if errors:
            return False, errors
        
        # Проверяем типы данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} is not numeric")
        
        # Проверяем на NaN значения
        for col in numeric_columns:
            if df[col].isna().any():
                errors.append(f"Column {col} contains NaN values")
        
        # Проверяем логику OHLC
        if not (df['high'] >= df['low']).all():
            errors.append("High price is not always >= Low price")
        
        if not (df['high'] >= df['open']).all():
            errors.append("High price is not always >= Open price")
        
        if not (df['high'] >= df['close']).all():
            errors.append("High price is not always >= Close price")
        
        if not (df['low'] <= df['open']).all():
            errors.append("Low price is not always <= Open price")
        
        if not (df['low'] <= df['close']).all():
            errors.append("Low price is not always <= Close price")
        
        # Проверяем на отрицательные цены
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                errors.append(f"Column {col} contains non-positive values")
        
        # Проверяем на отрицательные объемы
        if (df['volume'] < 0).any():
            errors.append("Volume contains negative values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_indicators(df: pd.DataFrame, feature_list: List[str]) -> Tuple[bool, List[str]]:
        """Валидация технических индикаторов"""
        errors = []
        
        # Проверяем наличие всех признаков
        missing_features = [feat for feat in feature_list if feat not in df.columns]
        if missing_features:
            errors.append(f"Missing features: {missing_features}")
        
        if errors:
            return False, errors
        
        # Проверяем на бесконечные значения
        for feature in feature_list:
            if df[feature].isin([np.inf, -np.inf]).any():
                errors.append(f"Feature {feature} contains infinite values")
        
        # Проверяем на слишком большие значения (возможные ошибки)
        for feature in feature_list:
            if df[feature].abs().max() > 1e10:
                errors.append(f"Feature {feature} contains extremely large values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_trading_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация торговых параметров"""
        errors = []
        
        # Проверяем процентные параметры
        percentage_params = ['sl_pct', 'tp_pct', 'fee', 'max_drawdown']
        for param in percentage_params:
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    errors.append(f"Parameter {param} must be between 0 and 1, got {value}")
        
        # Проверяем размер окна
        if 'window_size' in config:
            window_size = config['window_size']
            if not isinstance(window_size, int) or window_size <= 0:
                errors.append(f"window_size must be positive integer, got {window_size}")
        
        # Проверяем начальный депозит
        if 'initial_deposit' in config:
            deposit = config['initial_deposit']
            if not isinstance(deposit, (int, float)) or deposit <= 0:
                errors.append(f"initial_deposit must be positive number, got {deposit}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_model_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация параметров модели"""
        errors = []
        
        # Проверяем learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                errors.append(f"learning_rate must be between 0 and 1, got {lr}")
        
        # Проверяем размеры батчей и шагов
        batch_params = ['n_steps', 'batch_size', 'n_epochs']
        for param in batch_params:
            if param in config:
                value = config[param]
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"Parameter {param} must be positive integer, got {value}")
        
        # Проверяем архитектуру сети
        if 'net_arch' in config:
            net_arch = config['net_arch']
            if not isinstance(net_arch, list) or not all(isinstance(x, int) and x > 0 for x in net_arch):
                errors.append(f"net_arch must be list of positive integers, got {net_arch}")
        
        return len(errors) == 0, errors


class TradingValidator:
    """Валидатор торговых операций"""
    
    @staticmethod
    def validate_trade_action(action: int) -> bool:
        """Валидация торгового действия"""
        return action in [0, 1, 2]  # 0=hold, 1=buy, 2=sell
    
    @staticmethod
    def validate_position_size(size: float, balance: float, price: float, fee: float) -> Tuple[bool, str]:
        """Валидация размера позиции"""
        if size <= 0:
            return False, "Position size must be positive"
        
        if size > balance / (price * (1 + fee)):
            return False, "Insufficient balance for position"
        
        return True, ""
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Валидация цены"""
        return isinstance(price, (int, float)) and price > 0 and not np.isnan(price) and not np.isinf(price)


def validate_data_pipeline(df: pd.DataFrame, feature_list: List[str]) -> bool:
    """Полная валидация пайплайна данных"""
    logger.info("Starting data validation pipeline")
    
    # Валидация OHLCV данных
    is_valid_ohlcv, ohlcv_errors = DataValidator.validate_ohlcv_data(df)
    if not is_valid_ohlcv:
        logger.error(f"OHLCV validation failed: {ohlcv_errors}")
        return False
    
    # Валидация индикаторов
    is_valid_indicators, indicator_errors = DataValidator.validate_indicators(df, feature_list)
    if not is_valid_indicators:
        logger.error(f"Indicators validation failed: {indicator_errors}")
        return False
    
    logger.info("Data validation pipeline completed successfully")
    return True
