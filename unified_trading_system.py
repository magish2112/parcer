#!/usr/bin/env python3
"""
Единая система торговли без дублирования
Объединяет лучшие части всех существующих компонентов
"""

import os
import time
import json
import logging
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Конфигурация торговых параметров"""
    # Основные параметры
    initial_balance: float = 80000.0
    max_position_size: float = 0.2  # 20% вместо 50%
    transaction_fee: float = 0.0006
    lookback_window: int = 20
    
    # Улучшенные параметры трейлинга
    min_profit_threshold: float = 25.0
    trailing_tp_arm_threshold: float = 0.025
    tp_activate_pct: float = 0.1
    tp_trailing_pct: float = 0.04
    partial_tp_pct: float = 0.15
    partial_close_ratio: float = 0.25
    trailing_mult: float = 3.0
    
    # Адаптивный риск-скоринг
    max_consecutive_stops: int = 3
    max_partial_closes: int = 5
    min_trade_delay: int = 2

@dataclass
class ModelConfig:
    """Конфигурация модели PPO"""
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [512, 256, 128])

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "warmup", "timesteps": 50000, "learning_rate": 3e-4, "description": "Разогрев модели"},
        {"name": "exploration", "timesteps": 100000, "learning_rate": 1e-4, "description": "Исследование пространства"},
        {"name": "exploitation", "timesteps": 150000, "learning_rate": 5e-5, "description": "Эксплуатация знаний"},
        {"name": "fine_tuning", "timesteps": 100000, "learning_rate": 1e-5, "description": "Тонкая настройка"}
    ])
    target_reward: float = 200000.0
    eval_freq: int = 10000
    n_eval_episodes: int = 5

class UnifiedTradingEnv(gym.Env):
    """Единая улучшенная среда для торговли криптовалютами"""
    
    def __init__(
        self,
        df_path: str,
        config: TradingConfig,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Загрузка данных
        self.df = pd.read_csv(df_path)
        self.df = self.df.dropna().reset_index(drop=True)
        
        # Параметры среды
        self.initial_balance = config.initial_balance
        self.balance = config.initial_balance
        self.max_position_size = config.max_position_size
        self.fee = config.transaction_fee
        self.lookback_window = config.lookback_window
        
        # Торговые параметры
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.total_profit = 1.0
        self.trades = []
        self.current_step = 0
        self.averages_count = 0
        self.fibo_used = [False, False]
        self.last_trade_step = -100
        
        # Адаптивный риск-скоринг
        self.consecutive_stops = 0
        self.consecutive_wins = 0
        self.position_size_multiplier = 1.0
        
        # Улучшенные параметры трейлинга
        self.trailing_tp_active = False
        self.max_price_since_tp = 0.0
        self.trailing_tp_price = 0.0
        self.trailing_stop_price = None
        self.partial_closes_count = 0
        
        # Пространства действий и наблюдений
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback_window * 19 + 10,), dtype=np.float32
        )
        
        # Инициализация
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.total_profit = 1.0
        self.trades = []
        self.current_step = 0
        self.averages_count = 0
        self.fibo_used = [False, False]
        self.last_trade_step = -100
        
        # Сброс адаптивного риск-скоринга
        self.consecutive_stops = 0
        self.consecutive_wins = 0
        self.position_size_multiplier = 1.0
        
        # Сброс трейлинга
        self.trailing_tp_active = False
        self.max_price_since_tp = 0.0
        self.trailing_tp_price = 0.0
        self.trailing_stop_price = None
        self.partial_closes_count = 0
        
        return self._get_observation(), {}
    
    def _get_adaptive_position_size(self) -> float:
        """Адаптивный размер позиции на основе истории торговли"""
        base_size = self.max_position_size * self.position_size_multiplier
        
        # Дополнительное снижение при множественных стопах
        if self.consecutive_stops >= 2:
            base_size *= 0.5
        elif self.consecutive_stops >= 1:
            base_size *= 0.75
        
        # Увеличиваем размер при успешных сериях
        if self.consecutive_wins >= 3:
            base_size *= 1.2
        elif self.consecutive_wins >= 2:
            base_size *= 1.1
        
        return max(0.05, min(base_size, self.max_position_size))
    
    def _update_risk_scoring(self, trade_profit: float):
        """Обновление риск-скоринга на основе результата сделки"""
        if trade_profit > 0:
            self.consecutive_wins += 1
            self.consecutive_stops = 0
            
            if self.consecutive_wins >= 2:
                self.position_size_multiplier = min(1.0, self.position_size_multiplier + 0.1)
        else:
            self.consecutive_stops += 1
            self.consecutive_wins = 0
            
            if self.consecutive_stops >= 1:
                self.position_size_multiplier = max(0.3, self.position_size_multiplier - 0.2)
    
    def _get_observation(self) -> np.ndarray:
        """Получение наблюдения"""
        # Всегда возвращаем фиксированный размер
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        if self.current_step >= self.lookback_window:
            # Технические индикаторы
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step + 1
            
            # Основные данные
            prices = self.df['close'].iloc[start_idx:end_idx].values
            volumes = self.df['volume'].iloc[start_idx:end_idx].values
            highs = self.df['high'].iloc[start_idx:end_idx].values
            lows = self.df['low'].iloc[start_idx:end_idx].values
            
            # Технические индикаторы
            rsi = self.df['rsi'].iloc[start_idx:end_idx].values
            macd = self.df['macd'].iloc[start_idx:end_idx].values
            ema_14 = self.df['ema_14'].iloc[start_idx:end_idx].values
            ema200_d1 = self.df['ema200_d1'].iloc[start_idx:end_idx].values
            bb_bbm = self.df['bb_bbm'].iloc[start_idx:end_idx].values
            atr = self.df['atr'].iloc[start_idx:end_idx].values
            adx = self.df['adx'].iloc[start_idx:end_idx].values
            cci = self.df['cci'].iloc[start_idx:end_idx].values
            roc = self.df['roc'].iloc[start_idx:end_idx].values
            stoch = self.df['stoch'].iloc[start_idx:end_idx].values
            crsi = self.df['crsi'].iloc[start_idx:end_idx].values
            trend_ema = self.df['trend_ema'].iloc[start_idx:end_idx].values
            support = self.df['support'].iloc[start_idx:end_idx].values
            resistance = self.df['resistance'].iloc[start_idx:end_idx].values
            volume_spike = self.df['volume_spike'].iloc[start_idx:end_idx].values
            
            # Объединяем все признаки
            features = np.concatenate([
                prices, volumes, highs, lows, rsi, macd, ema_14, ema200_d1,
                bb_bbm, atr, adx, cci, roc, stoch, crsi, trend_ema,
                support, resistance, volume_spike
            ])
            
            # Нормализация
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Состояние портфеля
            portfolio_state = np.array([
                self.balance / self.initial_balance,
                self.position_size,
                self.avg_entry_price / self.df['close'].iloc[self.current_step] if self.position_size > 0 else 0,
                self.total_profit,
                len(self.trades),
                self.consecutive_stops / self.config.max_consecutive_stops,
                self.consecutive_wins / 5.0,
                self.position_size_multiplier,
                self.partial_closes_count / self.config.max_partial_closes,
                self.current_step / len(self.df)
            ], dtype=np.float32)
            
            # Заполняем наблюдение данными (обрезаем до нужного размера)
            max_features_size = self.observation_space.shape[0] - len(portfolio_state)
            features_trimmed = features[:max_features_size] if len(features) > max_features_size else features
            
            obs[:len(features_trimmed)] = features_trimmed
            obs[max_features_size:max_features_size+len(portfolio_state)] = portfolio_state
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполнение шага"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        self.current_step += 1
        price = self.df['close'].iloc[self.current_step]
        
        # Получение тренда
        trend_long = self.df['trend_ema'].iloc[self.current_step] > 0
        
        # Минимальная задержка между сделками
        if self.current_step - self.last_trade_step < self.config.min_trade_delay:
            trend_long = False
        
        profit = 0
        reward = 0
        
        # === УЛУЧШЕННАЯ ЛОГИКА ТРЕЙЛИНГА ===
        
        # Активация трейлинг-тейк-профита
        if self.position_size > 0 and not self.trailing_tp_active:
            arm_threshold = self.avg_entry_price * (1 + self.config.trailing_tp_arm_threshold)
            if price >= arm_threshold:
                if price >= self.avg_entry_price * (1 + self.config.tp_activate_pct):
                    self.trailing_tp_active = True
                    self.max_price_since_tp = price
                    self.trailing_tp_price = price * (1 - self.config.tp_trailing_pct)
                    logger.info(f"[{self.current_step}] Активирован трейлинг-тейк-профит: {self.trailing_tp_price:.2f}")
        
        # Обновление трейлинг-тейк-профита
        if self.position_size > 0 and self.trailing_tp_active:
            if price > self.max_price_since_tp:
                self.max_price_since_tp = price
                self.trailing_tp_price = price * (1 - self.config.tp_trailing_pct)
                logger.info(f"[{self.current_step}] Обновлён трейлинг-тейк-профит: {self.trailing_tp_price:.2f}")
            
            # Срабатывание трейлинг-тейк-профита
            if price <= self.trailing_tp_price:
                profit = self.position_size * (price * (1 - self.fee) - self.avg_entry_price * (1 + self.fee))
                
                if profit >= self.config.min_profit_threshold:
                    self.balance += self.position_size * price * (1 - self.fee)
                    self.position_size = 0
                    self.averages_count = 0
                    self.fibo_used = [False, False]
                    self.last_trade_step = self.current_step
                    self.trailing_stop_price = None
                    self.trailing_tp_active = False
                    self.partial_closes_count = 0
                    reward += profit
                    self.total_profit *= (1 + profit / self.initial_balance)
                    self.trades.append(profit)
                    self._update_risk_scoring(profit)
                    logger.info(f"[{self.current_step}] Трейлинг-тейк-профит по {price:.2f}, профит: {profit:.5f} USDT")
                else:
                    logger.info(f"[{self.current_step}] Трейлинг-тейк-профит отменен: прибыль {profit:.5f} USDT < {self.config.min_profit_threshold} USDT")
                    return self._get_observation(), reward, False, False, {}
        
        # === УЛУЧШЕННОЕ ЧАСТИЧНОЕ ЗАКРЫТИЕ ===
        if (self.position_size > 0 and 
            price >= self.avg_entry_price * (1 + self.config.partial_tp_pct) and
            self.partial_closes_count < self.config.max_partial_closes):
            
            close_size = self.position_size * self.config.partial_close_ratio
            profit = close_size * (price * (1 - self.fee) - self.avg_entry_price * (1 + self.fee))
            
            if profit >= self.config.min_profit_threshold:
                self.balance += close_size * price * (1 - self.fee)
                self.position_size -= close_size
                self.partial_closes_count += 1
                reward += profit
                self.total_profit *= (1 + profit / self.initial_balance)
                self.trades.append(profit)
                logger.info(f"[{self.current_step}] Частичное закрытие {self.config.partial_close_ratio*100:.0f}% по {price:.2f}, профит: {profit:.5f} USDT")
            else:
                logger.info(f"[{self.current_step}] Частичное закрытие отменено: прибыль {profit:.5f} USDT < {self.config.min_profit_threshold} USDT")
                return self._get_observation(), reward, False, False, {}
        
        # === УЛУЧШЕННЫЙ СТОП-ЛОСС ===
        if self.position_size > 0:
            atr = self.df['atr'].iloc[self.current_step]
            trailing_stop = price - self.config.trailing_mult * atr
            
            if self.trailing_stop_price is None:
                self.trailing_stop_price = trailing_stop
            else:
                self.trailing_stop_price = max(self.trailing_stop_price, trailing_stop)
            
            if price <= self.trailing_stop_price:
                profit = self.position_size * (price * (1 - self.fee) - self.avg_entry_price * (1 + self.fee))
                self.balance += self.position_size * price * (1 - self.fee)
                self.position_size = 0
                self.averages_count = 0
                self.fibo_used = [False, False]
                self.last_trade_step = self.current_step
                self.trailing_stop_price = None
                self.trailing_tp_active = False
                self.partial_closes_count = 0
                reward += profit
                self.total_profit *= (1 + profit / self.initial_balance)
                self.trades.append(profit)
                self._update_risk_scoring(profit)
                logger.info(f"[{self.current_step}] Стоп-лосс сработал по {price:.2f}, убыток: {profit:.5f} USDT")
        
        # === ОТКРЫТИЕ ПОЗИЦИЙ С АДАПТИВНЫМ РАЗМЕРОМ ===
        if action == 1 and trend_long and self.position_size == 0:
            adaptive_size = self._get_adaptive_position_size()
            position_value = self.balance * adaptive_size
            
            if position_value >= price * 0.01:
                self.position_size = position_value / price
                self.avg_entry_price = price
                self.balance -= position_value
                self.averages_count = 1
                self.last_trade_step = self.current_step
                self.trailing_stop_price = None
                self.trailing_tp_active = False
                self.partial_closes_count = 0
                logger.info(f"[{self.current_step}] Открыта позиция: {adaptive_size*100:.1f}% от баланса, size={self.position_size:.6f} @ {price:.2f}")
        
        # === ПИРАМИДИНГ С ОГРАНИЧЕНИЯМИ ===
        elif action == 1 and trend_long and self.position_size > 0 and self.averages_count < 2:
            adaptive_size = self._get_adaptive_position_size() * 0.5
            position_value = self.balance * adaptive_size
            
            if position_value >= price * 0.01:
                new_size = position_value / price
                total_cost = self.position_size * self.avg_entry_price + new_size * price
                self.avg_entry_price = total_cost / (self.position_size + new_size)
                self.position_size += new_size
                self.balance -= position_value
                self.averages_count += 1
                self.last_trade_step = self.current_step
                logger.info(f"[{self.current_step}] Пирамидинг #{self.averages_count}: {adaptive_size*100:.1f}% от баланса, size={new_size:.6f} @ {price:.2f}")
        
        # === ПРОДАЖА ===
        elif action == 2 and self.position_size > 0:
            profit = self.position_size * (price * (1 - self.fee) - self.avg_entry_price * (1 + self.fee))
            self.balance += self.position_size * price * (1 - self.fee)
            self.position_size = 0
            self.averages_count = 0
            self.fibo_used = [False, False]
            self.last_trade_step = self.current_step
            self.trailing_stop_price = None
            self.trailing_tp_active = False
            self.partial_closes_count = 0
            reward += profit
            self.total_profit *= (1 + profit / self.initial_balance)
            self.trades.append(profit)
            self._update_risk_scoring(profit)
            logger.info(f"[{self.current_step}] Продажа по {price:.2f}, профит: {profit:.5f} USDT")
        
        # Награда за удержание позиции
        if self.position_size > 0:
            unrealized_profit = self.position_size * (price - self.avg_entry_price)
            reward += unrealized_profit * 0.001
        
        # Штраф за частые сделки
        if self.current_step - self.last_trade_step < 5:
            reward -= 0.1
        
        # Информация
        info = {
            'balance': self.balance,
            'position_size': self.position_size,
            'total_profit': self.total_profit,
            'consecutive_stops': self.consecutive_stops,
            'consecutive_wins': self.consecutive_wins,
            'position_size_multiplier': self.position_size_multiplier,
            'partial_closes_count': self.partial_closes_count
        }
        
        return self._get_observation(), reward, False, False, info
    
    def render(self):
        """Визуализация"""
        if self.render_mode == 'human':
            logger.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position_size:.6f}, Profit: {self.total_profit:.4f}")
    
    def close(self):
        """Закрытие среды"""
        pass

class UnifiedTrainingManager:
    """Единый менеджер обучения без дублирования"""
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        # Загрузка конфигурации
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Фильтруем только нужные параметры для TradingConfig
            trading_params = config_data.get('trading', {})
            filtered_params = {}
            
            # Маппинг параметров
            if 'initial_deposit' in trading_params:
                filtered_params['initial_balance'] = trading_params['initial_deposit']
            if 'fee' in trading_params:
                filtered_params['transaction_fee'] = trading_params['fee']
            if 'window_size' in trading_params:
                filtered_params['lookback_window'] = trading_params['window_size']
            
            self.trading_config = TradingConfig(**filtered_params)
            
            # Фильтруем параметры для ModelConfig
            model_params = config_data.get('model', {})
            filtered_model_params = {k: v for k, v in model_params.items() 
                                   if k in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 
                                           'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 
                                           'vf_coef', 'max_grad_norm', 'net_arch']}
            self.model_config = ModelConfig(**filtered_model_params)
            
            # TrainingConfig используем как есть
            self.training_config = TrainingConfig()
        except FileNotFoundError:
            # Используем конфигурацию по умолчанию
            logger.warning(f"Файл конфигурации {config_path} не найден, используем настройки по умолчанию")
            self.trading_config = TradingConfig()
            self.model_config = ModelConfig()
            self.training_config = TrainingConfig()
        
        self.training_dir = Path("unified_training")
        self.training_dir.mkdir(exist_ok=True)
        
        self.current_stage = 0
        self.best_reward = -np.inf
        self.training_history = []
    
    def create_environment(self, stage_config: Dict[str, Any]) -> UnifiedTradingEnv:
        """Создание среды для этапа"""
        # Создаем копию конфигурации для каждого этапа
        stage_config_copy = TradingConfig(
            initial_balance=self.trading_config.initial_balance,
            max_position_size=self.trading_config.max_position_size,
            transaction_fee=self.trading_config.transaction_fee,
            lookback_window=self.trading_config.lookback_window,
            min_profit_threshold=self.trading_config.min_profit_threshold,
            trailing_tp_arm_threshold=self.trading_config.trailing_tp_arm_threshold,
            tp_activate_pct=self.trading_config.tp_activate_pct,
            tp_trailing_pct=self.trading_config.tp_trailing_pct,
            partial_tp_pct=self.trading_config.partial_tp_pct,
            partial_close_ratio=self.trading_config.partial_close_ratio,
            trailing_mult=self.trading_config.trailing_mult,
            max_consecutive_stops=self.trading_config.max_consecutive_stops,
            max_partial_closes=self.trading_config.max_partial_closes,
            min_trade_delay=self.trading_config.min_trade_delay
        )
        
        # Адаптивные параметры для разных этапов
        if stage_config["name"] == "warmup":
            stage_config_copy.min_profit_threshold = 30.0
            stage_config_copy.trailing_tp_arm_threshold = 0.03
        elif stage_config["name"] == "exploration":
            stage_config_copy.min_profit_threshold = 25.0
            stage_config_copy.trailing_tp_arm_threshold = 0.025
        elif stage_config["name"] == "exploitation":
            stage_config_copy.min_profit_threshold = 20.0
            stage_config_copy.trailing_tp_arm_threshold = 0.02
        else:  # fine_tuning
            stage_config_copy.min_profit_threshold = 15.0
            stage_config_copy.trailing_tp_arm_threshold = 0.015
        
        return UnifiedTradingEnv(
            df_path="btc_4h_full_fixed.csv",
            config=stage_config_copy
        )
    
    def create_model(self, stage_config: Dict[str, Any], env) -> PPO:
        """Создание модели PPO"""
        model_params = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": stage_config["learning_rate"],
            "n_steps": self.model_config.n_steps,
            "batch_size": self.model_config.batch_size,
            "n_epochs": self.model_config.n_epochs,
            "gamma": self.model_config.gamma,
            "gae_lambda": self.model_config.gae_lambda,
            "clip_range": self.model_config.clip_range,
            "ent_coef": self.model_config.ent_coef,
            "vf_coef": self.model_config.vf_coef,
            "max_grad_norm": self.model_config.max_grad_norm,
            "verbose": 1,
            "device": "auto"
        }
        
        return PPO(**model_params)
    
    def train_stage(self, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Обучение на одном этапе"""
        logger.info(f"Начинаем этап: {stage_config['name']} - {stage_config['description']}")
        
        # Создание среды
        env = self.create_environment(stage_config)
        vec_env = DummyVecEnv([lambda: env])
        
        # Создание модели
        model = self.create_model(stage_config, vec_env)
        
        # Без коллбэков для упрощения
        eval_callback = None
        
        # Обучение
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=stage_config["timesteps"],
                progress_bar=False
            )
        except KeyboardInterrupt:
            logger.info("Обучение прервано пользователем")
        
        training_time = time.time() - start_time
        
        # Оценка модели
        mean_reward, std_reward = self.evaluate_model(model, vec_env, n_eval_episodes=self.training_config.n_eval_episodes)
        
        metrics = {
            "stage": stage_config["name"],
            "timesteps": stage_config["timesteps"],
            "learning_rate": stage_config["learning_rate"],
            "training_time": training_time,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_history.append(metrics)
        
        # Сохранение чекпоинта
        self.save_checkpoint(model, stage_config["name"], metrics)
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            logger.info(f"Новый рекорд награды: {mean_reward:.2f}")
        
        logger.info(f"Этап {stage_config['name']} завершен. Награда: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Очистка памяти
        del model, vec_env, env
        
        return metrics
    
    def evaluate_model(self, model: PPO, env, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """Оценка модели"""
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0] if hasattr(reward, '__len__') else reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def save_checkpoint(self, model: PPO, stage_name: str, metrics: Dict[str, Any]):
        """Сохранение чекпоинта"""
        checkpoint_dir = self.training_dir / f"checkpoint_{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        model.save(checkpoint_dir / "model")
        
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Чекпоинт сохранен: {checkpoint_dir}")
        return checkpoint_dir
    
    def run_training(self):
        """Запуск полного цикла обучения"""
        logger.info("Начинаем единое обучение без дублирования")
        logger.info(f"Планируется {len(self.training_config.stages)} этапов")
        
        total_start_time = time.time()
        
        for i, stage_config in enumerate(self.training_config.stages):
            self.current_stage = i
            stage_metrics = self.train_stage(stage_config)
            
            # Сохранение истории обучения
            with open(self.training_dir / "training_history.json", "w") as f:
                json.dump(self.training_history, f, indent=2)
            
            # Проверка на раннюю остановку
            if stage_metrics["mean_reward"] > self.training_config.target_reward * 1.25:
                logger.info("Достигнута целевая награда, завершаем обучение")
                break
        
        total_time = time.time() - total_start_time
        logger.info(f"Обучение завершено за {total_time/3600:.2f} часов")
        logger.info(f"Лучшая награда: {self.best_reward:.2f}")

def main():
    """Главная функция"""
    try:
        trainer = UnifiedTrainingManager()
        trainer.run_training()
    except Exception as e:
        logger.error(f"Ошибка в обучении: {e}")
        raise

if __name__ == "__main__":
    main()
