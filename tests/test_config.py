#!/usr/bin/env python3
"""
Тесты для системы конфигурации
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.config import Config, TradingConfig, ModelConfig, DataConfig, get_config


class TestTradingConfig:
    """Тесты для TradingConfig"""
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        config = TradingConfig()
        
        assert config.symbol == 'BTCUSDT'
        assert config.initial_deposit == 10000.0
        assert config.window_size == 24
        assert config.sl_pct == 0.03
        assert config.tp_pct == 0.07
        assert config.fee == 0.001
    
    def test_custom_values(self):
        """Тест пользовательских значений"""
        config = TradingConfig(
            symbol='ETHUSDT',
            initial_deposit=5000.0,
            window_size=48
        )
        
        assert config.symbol == 'ETHUSDT'
        assert config.initial_deposit == 5000.0
        assert config.window_size == 48


class TestModelConfig:
    """Тесты для ModelConfig"""
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        config = ModelConfig()
        
        assert config.learning_rate == 1e-4
        assert config.n_steps == 4096
        assert config.batch_size == 256
        assert config.n_epochs == 10
        assert config.gamma == 0.99
    
    def test_network_architecture(self):
        """Тест архитектуры сети"""
        config = ModelConfig()
        
        assert len(config.net_arch) == 3
        assert config.net_arch == [512, 256, 128]
        assert config.activation_fn == 'ReLU'


class TestDataConfig:
    """Тесты для DataConfig"""
    
    def test_default_values(self):
        """Тест значений по умолчанию"""
        config = DataConfig()
        
        assert config.dataset_path == 'btc_4h_full_fixed.csv'
        assert config.interval == '240'
        assert config.limit == 200
    
    def test_feature_list(self):
        """Тест списка признаков"""
        config = DataConfig()
        
        assert len(config.feature_list) == 22
        assert 'close' in config.feature_list
        assert 'volume' in config.feature_list
        assert 'rsi' in config.feature_list


class TestConfig:
    """Тесты для главной конфигурации"""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = Config()
        
        assert isinstance(config.trading, TradingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
    
    def test_yaml_loading(self):
        """Тест загрузки из YAML"""
        # Создаем временный YAML файл
        yaml_content = """
trading:
  symbol: 'ETHUSDT'
  initial_deposit: 5000.0
  window_size: 48

model:
  learning_rate: 2e-4
  n_steps: 2048

data:
  dataset_path: 'eth_4h_data.csv'
  interval: '60'
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = Config.from_yaml(temp_path)
            
            assert config.trading.symbol == 'ETHUSDT'
            assert config.trading.initial_deposit == 5000.0
            assert config.trading.window_size == 48
            assert config.model.learning_rate == 2e-4
            assert config.model.n_steps == 2048
            assert config.data.dataset_path == 'eth_4h_data.csv'
            assert config.data.interval == '60'
            
        finally:
            os.unlink(temp_path)
    
    def test_yaml_saving(self):
        """Тест сохранения в YAML"""
        config = Config()
        config.trading.symbol = 'ETHUSDT'
        config.model.learning_rate = 2e-4
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_yaml(temp_path)
            
            # Загружаем обратно и проверяем
            loaded_config = Config.from_yaml(temp_path)
            assert loaded_config.trading.symbol == 'ETHUSDT'
            assert loaded_config.model.learning_rate == 2e-4
            
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_file(self):
        """Тест загрузки несуществующего файла"""
        config = Config.from_yaml('nonexistent.yaml')
        
        # Должна вернуться конфигурация по умолчанию
        assert config.trading.symbol == 'BTCUSDT'
        assert config.model.learning_rate == 1e-4


class TestGetConfig:
    """Тесты для функции get_config"""
    
    def test_get_config_default(self):
        """Тест получения конфигурации по умолчанию"""
        config = get_config()
        
        assert isinstance(config, Config)
        assert config.trading.symbol == 'BTCUSDT'
    
    def test_get_config_custom_path(self):
        """Тест получения конфигурации с пользовательским путем"""
        # Создаем временный конфиг
        yaml_content = """
trading:
  symbol: 'ADAUSDT'
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = get_config(temp_path)
            assert config.trading.symbol == 'ADAUSDT'
            
        finally:
            os.unlink(temp_path)
