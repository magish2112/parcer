#!/usr/bin/env python3
"""
Единая система конфигурации для торгового бота
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class TradingConfig:
    symbol: str = 'BTCUSDT'
    initial_deposit: float = 10000.0
    window_size: int = 24
    sl_pct: float = 0.03
    tp_pct: float = 0.07
    fee: float = 0.001
    taker_fee: float = 0.0006  # Bybit taker fee 0.06%
    min_trade_delay: int = 1
    max_drawdown: float = 0.2
    
    # Параметры пирамидинга
    pyramid_levels: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.50])
    pyramid_drawdown_thresholds: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.25])
    
    # Параметры трейлинг-стопа
    trailing_tp_activate_pct: float = 0.05
    trailing_tp_trailing_pct: float = 0.022
    trailing_stop_multiplier: float = 1.5
    
    # Параметры частичного закрытия
    partial_tp_pct: float = 0.08
    partial_close_ratio: float = 0.5
    
    # Новые улучшенные параметры
    min_profit_threshold: float = 15.0
    trailing_tp_arm_threshold: float = 0.015
    dynamic_sl_enabled: bool = True
    sl_atr_multiplier: float = 2.0


@dataclass
class ModelConfig:
    """Конфигурация модели PPO"""
    learning_rate: float = 1e-4
    n_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Архитектура сети
    net_arch: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation_fn: str = 'ReLU'
    
    # Параметры обучения
    total_timesteps: int = 1000000
    learning_episodes: int = 10000
    min_data_points: int = 1000


@dataclass
class DataConfig:
    """Конфигурация данных"""
    dataset_path: str = 'btc_4h_full_fixed.csv'
    interval: str = '240'  # 4H
    limit: int = 200
    
    # Список признаков (только доступные в данных)
    feature_list: List[str] = field(default_factory=lambda: [
        'close', 'volume', 'rsi', 'ema_14', 'macd', 'atr', 'bb_bbm', 'adx', 'cci', 'roc', 'stoch',
        'crsi', 'sideways_volume', 'dist_to_support', 'dist_to_resistance', 'false_breakout',
        'volume_spike', 'trend_ema', 'support', 'resistance', 'ema200_d1', 'atr_mult_trailing_stop'
    ])
    
    # Параметры индикаторов
    rsi_window: int = 14
    ema_window: int = 14
    atr_window: int = 14
    adx_window: int = 14
    cci_window: int = 20
    roc_window: int = 12
    bb_window: int = 20
    support_resistance_window: int = 50


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = 'INFO'
    log_dir: str = 'logs'
    log_file: str = 'trading.log'
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


@dataclass
class APIConfig:
    """Конфигурация API"""
    testnet: bool = True
    api_key_env: str = 'BYBIT_API_KEY'
    api_secret_env: str = 'BYBIT_API_SECRET'
    timeout: int = 30
    retry_count: int = 3


@dataclass
class MonitoringConfig:
    """Конфигурация мониторинга"""
    enabled: bool = True
    metrics_interval: int = 100  # шагов
    save_interval: int = 1000    # шагов
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.15,
        'min_win_rate': 0.4,
        'max_loss_streak': 10
    })


@dataclass
class Config:
    """Главная конфигурация"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config = cls()
            
            # Обновляем конфигурации из YAML
            if 'trading' in data:
                for key, value in data['trading'].items():
                    if hasattr(config.trading, key):
                        setattr(config.trading, key, value)
            
            if 'model' in data:
                for key, value in data['model'].items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)
            
            if 'data' in data:
                for key, value in data['data'].items():
                    if hasattr(config.data, key):
                        setattr(config.data, key, value)
            
            if 'logging' in data:
                for key, value in data['logging'].items():
                    if hasattr(config.logging, key):
                        setattr(config.logging, key, value)
            
            if 'api' in data:
                for key, value in data['api'].items():
                    if hasattr(config.api, key):
                        setattr(config.api, key, value)
            
            if 'monitoring' in data:
                for key, value in data['monitoring'].items():
                    if hasattr(config.monitoring, key):
                        setattr(config.monitoring, key, value)
            
            return config
            
        except FileNotFoundError:
            print(f"Config file {path} not found, using defaults")
            return cls()
        except yaml.YAMLError as e:
            print(f"Error parsing config file {path}: {e}")
            return cls()
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()
    
    def save_yaml(self, path: str) -> None:
        """Сохранение конфигурации в YAML файл"""
        try:
            data = {
                'trading': {
                    'symbol': self.trading.symbol,
                    'initial_deposit': self.trading.initial_deposit,
                    'window_size': self.trading.window_size,
                    'sl_pct': self.trading.sl_pct,
                    'tp_pct': self.trading.tp_pct,
                    'fee': self.trading.fee,
                    'min_trade_delay': self.trading.min_trade_delay,
                    'max_drawdown': self.trading.max_drawdown,
                    'pyramid_levels': self.trading.pyramid_levels,
                    'pyramid_drawdown_thresholds': self.trading.pyramid_drawdown_thresholds,
                    'trailing_tp_activate_pct': self.trading.trailing_tp_activate_pct,
                    'trailing_tp_trailing_pct': self.trading.trailing_tp_trailing_pct,
                    'trailing_stop_multiplier': self.trading.trailing_stop_multiplier,
                    'partial_tp_pct': self.trading.partial_tp_pct,
                    'partial_close_ratio': self.trading.partial_close_ratio,
                },
                'model': {
                    'learning_rate': self.model.learning_rate,
                    'n_steps': self.model.n_steps,
                    'batch_size': self.model.batch_size,
                    'n_epochs': self.model.n_epochs,
                    'gamma': self.model.gamma,
                    'gae_lambda': self.model.gae_lambda,
                    'clip_range': self.model.clip_range,
                    'ent_coef': self.model.ent_coef,
                    'vf_coef': self.model.vf_coef,
                    'max_grad_norm': self.model.max_grad_norm,
                    'net_arch': self.model.net_arch,
                    'activation_fn': self.model.activation_fn,
                    'total_timesteps': self.model.total_timesteps,
                    'learning_episodes': self.model.learning_episodes,
                    'min_data_points': self.model.min_data_points,
                },
                'data': {
                    'dataset_path': self.data.dataset_path,
                    'interval': self.data.interval,
                    'limit': self.data.limit,
                    'feature_list': self.data.feature_list,
                    'rsi_window': self.data.rsi_window,
                    'ema_window': self.data.ema_window,
                    'atr_window': self.data.atr_window,
                    'adx_window': self.data.adx_window,
                    'cci_window': self.data.cci_window,
                    'roc_window': self.data.roc_window,
                    'bb_window': self.data.bb_window,
                    'support_resistance_window': self.data.support_resistance_window,
                },
                'logging': {
                    'level': self.logging.level,
                    'log_dir': self.logging.log_dir,
                    'log_file': self.logging.log_file,
                    'max_file_size': self.logging.max_file_size,
                    'backup_count': self.logging.backup_count,
                    'format': self.logging.format,
                },
                'api': {
                    'testnet': self.api.testnet,
                    'api_key_env': self.api.api_key_env,
                    'api_secret_env': self.api.api_secret_env,
                    'timeout': self.api.timeout,
                    'retry_count': self.api.retry_count,
                },
                'monitoring': {
                    'enabled': self.monitoring.enabled,
                    'metrics_interval': self.monitoring.metrics_interval,
                    'save_interval': self.monitoring.save_interval,
                    'alert_thresholds': self.monitoring.alert_thresholds,
                }
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                
        except Exception as e:
            print(f"Error saving config: {e}")


def get_config(config_path: str = 'config/trading_config.yaml') -> Config:
    """Получение конфигурации"""
    return Config.from_yaml(config_path)


def get_api_credentials(config: Config) -> tuple[str, str]:
    """Безопасное получение API ключей"""
    api_key = os.getenv(config.api.api_key_env)
    api_secret = os.getenv(config.api.api_secret_env)
    
    if not api_key or not api_secret:
        raise ValueError(f"API credentials not found in environment variables: "
                        f"{config.api.api_key_env}, {config.api.api_secret_env}")
    
    return api_key, api_secret
