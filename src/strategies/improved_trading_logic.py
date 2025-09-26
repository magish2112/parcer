#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.logging import get_logger


@dataclass
class ImprovedTradingConfig:
    
    # Базовые параметры
    symbol: str = 'BTCUSDT'
    initial_deposit: float = 10000.0
    window_size: int = 24
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%
    fee: float = 0.0006        # Для совместимости
    min_trade_delay: int = 1
    
    # Улучшенные параметры стоп-лосса
    sl_pct: float = 0.02  # Уменьшено с 0.03 для снижения убытков
    trailing_stop_multiplier: float = 1.5  # Уменьшено с 2.5 для более быстрого выхода
    # Отключаем динамический стоп-лосс по умолчанию, чтобы ранние откаты не выбивали позицию
    dynamic_sl_enabled: bool = False
    sl_atr_multiplier: float = 1.5  # Уменьшено с 2.0
    
    # Улучшенные параметры тейк-профита
    tp_pct: float = 0.07
    trailing_tp_activate_pct: float = 0.05  # Уменьшено с 0.08 для более ранней активации
    trailing_tp_trailing_pct: float = 0.025  # Уменьшено с 0.035 для более агрессивного трейлинга
    trailing_tp_arm_threshold: float = 0.015  # Новый параметр: активация только после покрытия комиссий
    
    # Параметры частичного закрытия
    partial_tp_pct: float = 0.12  # Увеличено с 0.08
    partial_close_ratio: float = 0.3  # Уменьшено с 0.5
    min_profit_threshold: float = 25.0  # Увеличено с 15.0 для фильтрации мелких сделок
    
    # Параметры пирамидинга
    pyramid_levels: list = None
    pyramid_drawdown_thresholds: list = None
    
    # Параметры волатильности
    volatility_lookback: int = 20
    high_volatility_threshold: float = 1.5
    low_volatility_threshold: float = 0.5
    min_volatility_for_trade: float = 0.1
    max_volatility_for_trade: float = 10.0
    trend_strength_min: float = 10.0
    rsi_oversold: float = 15.0
    rsi_overbought: float = 85.0
    volume_spike_threshold: float = 0.8
    
    def __post_init__(self):
        if self.pyramid_levels is None:
            # Уровни пирамидинга: 10 %, 20 %, 50 % от депозита
            self.pyramid_levels = [0.10, 0.20, 0.50]
        if self.pyramid_drawdown_thresholds is None:
            # Порог второй покупки — просадка ≥ 5 %, третьей — ≥ 8 %
            # (первый уровень открывается без учёта просадки)
            self.pyramid_drawdown_thresholds = [0.05, 0.08, 0.08]


class ImprovedTradingLogic:
    
    def __init__(self, config: ImprovedTradingConfig):
        self.config = config
        self.logger = get_logger("improved_trading")
        
        # Состояние позиции
        self.position_size = 0.0
        self.position_type = None  # 'long', 'short' или None
        self.avg_entry_price = 0.0
        self.entry_step = 0
        self.max_price_since_entry = 0.0
        self.min_price_since_entry = float('inf')
        
        # Трейлинг-стопы
        self.trailing_stop_price = None
        self.trailing_tp_active = False
        self.trailing_tp_price = None
        self.trailing_tp_armed = False
        
        # Частичные закрытия
        self.partial_closes_count = 0
        self.max_partial_closes = 2
        
        # Анализ волатильности
        self.volatility_history = []
        self.current_volatility = 1.0
        
        # Статистика
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
    
    def update_volatility(self, price_history: np.ndarray) -> float:
        if len(price_history) < self.config.volatility_lookback:
            return 1.0
        
        # Вычисляем волатильность как стандартное отклонение доходности
        returns = np.diff(price_history[-self.config.volatility_lookback:]) / price_history[-self.config.volatility_lookback:-1]
        volatility = np.std(returns) * np.sqrt(24)  # Нормализация для 4H данных
        
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
        
        self.current_volatility = np.mean(self.volatility_history[-10:]) if len(self.volatility_history) >= 10 else volatility
        return self.current_volatility
    
    def get_adaptive_parameters(self, atr: float, volatility: float) -> Dict[str, float]:
        
        # Адаптивный множитель для стоп-лосса
        if volatility > self.config.high_volatility_threshold:
            sl_multiplier = self.config.trailing_stop_multiplier * 1.5
            tp_multiplier = 1.2
        elif volatility < self.config.low_volatility_threshold:
            sl_multiplier = self.config.trailing_stop_multiplier * 0.8
            tp_multiplier = 0.9
        else:
            sl_multiplier = self.config.trailing_stop_multiplier
            tp_multiplier = 1.0
        
        return {
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier,
            'atr': atr,
            'volatility': volatility
        }
    
    def should_activate_trailing_tp(self, current_price: float) -> bool:
        if self.position_size <= 0:
            return False
        
        # Базовый порог активации
        base_threshold = self.avg_entry_price * (1 + self.config.trailing_tp_activate_pct)
        
        # Arm-порог: активация только после покрытия комиссий и спреда
        arm_threshold = self.avg_entry_price * (1 + self.config.trailing_tp_arm_threshold)
        
        # Проверяем arm-порог
        if not self.trailing_tp_armed and current_price >= arm_threshold:
            self.trailing_tp_armed = True
            self.logger.info(f"Trailing TP armed at {current_price:.2f}")
        
        # Активация только после arm-порога
        return self.trailing_tp_armed and current_price >= base_threshold
    
    def handle_trailing_take_profit(self, current_price: float, atr: float) -> Optional[float]:
        if self.position_size <= 0:
            return None
        
        # Активация трейлинг-тейк-профита
        if not self.trailing_tp_active:
            if self.should_activate_trailing_tp(current_price):
                self.trailing_tp_active = True
                self.max_price_since_tp = current_price
                
                # Адаптивный trailing процент
                adaptive_params = self.get_adaptive_parameters(atr, self.current_volatility)
                trailing_pct = self.config.trailing_tp_trailing_pct * adaptive_params['tp_multiplier']
                
                self.trailing_tp_price = current_price * (1 - trailing_pct)
                self.logger.info(f"Trailing TP activated at {self.trailing_tp_price:.2f} (trailing: {trailing_pct:.3f})")
        
        # Обновление трейлинг-тейк-профита
        if self.trailing_tp_active:
            if current_price > self.max_price_since_tp:
                self.max_price_since_tp = current_price
                
                # Адаптивный trailing процент
                adaptive_params = self.get_adaptive_parameters(atr, self.current_volatility)
                trailing_pct = self.config.trailing_tp_trailing_pct * adaptive_params['tp_multiplier']
                
                self.trailing_tp_price = current_price * (1 - trailing_pct)
            
            # Срабатывание трейлинг-тейк-профита
            if current_price <= self.trailing_tp_price:
                profit = self._close_position(current_price, 'sell_trailing_tp')
                self.logger.info(f"Trailing TP triggered: {profit:.2f} USDT")
                return profit
        
        return None
    
    def handle_partial_take_profit(self, current_price: float) -> Optional[float]:
        if self.position_size <= 0 or self.partial_closes_count >= self.max_partial_closes:
            return None
        
        # Проверяем минимальный порог прибыли
        potential_profit = current_price - self.avg_entry_price
        if potential_profit < self.config.min_profit_threshold:
            return None
        
        # Проверяем порог частичного тейк-профита
        if current_price >= self.avg_entry_price * (1 + self.config.partial_tp_pct):
            close_size = self.position_size * self.config.partial_close_ratio
            profit = close_size * (current_price * (1 - self.config.taker_fee) -
                                 self.avg_entry_price * (1 + self.config.taker_fee))
            
            # Проверяем, что прибыль превышает минимальный порог
            if profit >= self.config.min_profit_threshold:
                self.position_size -= close_size
                self.partial_closes_count += 1
                
                self.logger.info(f"Partial TP #{self.partial_closes_count}: {profit:.2f} USDT")
                return profit
        
        return None
    
    def handle_dynamic_stop_loss(self, current_price: float, atr: float) -> Optional[float]:
        if self.position_size <= 0:
            return None
        
        # Адаптивные параметры
        adaptive_params = self.get_adaptive_parameters(atr, self.current_volatility)
        sl_multiplier = adaptive_params['sl_multiplier']
        
        # Динамический стоп-лосс на основе ATR
        if self.config.dynamic_sl_enabled:
            dynamic_sl_price = self.avg_entry_price - (atr * sl_multiplier)
        else:
            dynamic_sl_price = self.avg_entry_price * (1 - self.config.sl_pct)
        
        # Обновляем trailing stop
        if self.trailing_stop_price is None:
            self.trailing_stop_price = dynamic_sl_price
        else:
            # Trailing stop только в сторону прибыли
            if dynamic_sl_price > self.trailing_stop_price:
                self.trailing_stop_price = dynamic_sl_price
        
        # Проверяем срабатывание стоп-лосса
        if current_price <= self.trailing_stop_price:
            profit = self._close_position(current_price, 'sell_stop_loss')
            self.logger.info(f"Stop loss triggered: {profit:.2f} USDT (SL: {self.trailing_stop_price:.2f})")
            return profit
        
        return None

    def check_entry_conditions(self, indicators: Dict[str, float], current_price: float, is_long: bool = True) -> bool:
        """
        Полностью отключены фильтры - модель может торговать всегда
        """
        return True

    def _calculate_current_volatility(self, indicators: Dict[str, float] = None) -> float:
        """Расчет текущей волатильности на основе ATR и других индикаторов"""
        try:
            if indicators is None:
                return 1.5  # Заглушка если индикаторы не переданы

            # Основная метрика волатильности - ATR
            atr = indicators.get('atr', 0.01)
            close_price = indicators.get('close', 50000.0)

            # Нормализованная волатильность (ATR в процентах от цены)
            volatility_pct = (atr / close_price) * 100

            # Дополнительные факторы волатильности
            bb_width = indicators.get('bb_width', 0.02)  # Ширина полос Боллинджера
            rsi = indicators.get('rsi', 50.0)

            # Комбинированная метрика волатильности
            combined_volatility = volatility_pct * (1 + bb_width) * (1 + abs(rsi - 50) / 50)

            # Ограничиваем диапазон
            combined_volatility = max(0.5, min(5.0, combined_volatility))

            return combined_volatility

        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
            return 1.5

    def _close_position(self, price: float, action_type: str) -> float:
        if self.position_type == 'long':
            # Для лонга: общая стоимость закрытия = размер * цена_продажи * (1 - комиссия)
            # Прибыль = стоимость закрытия - первоначальные инвестиции
            close_value = self.position_size * price * (1 - self.config.taker_fee)
            cost_basis = self.position_size * self.avg_entry_price * (1 + self.config.taker_fee)
            profit = close_value - cost_basis
        elif self.position_type == 'short':
            # Для шорта: общая стоимость закрытия = размер * цена_покупки * (1 - комиссия)
            # Прибыль = первоначальные инвестиции - стоимость закрытия
            close_value = self.position_size * price * (1 - self.config.taker_fee)
            cost_basis = self.position_size * self.avg_entry_price * (1 + self.config.taker_fee)
            profit = cost_basis - close_value
        else:
            profit = 0.0

        # Обновляем статистику
        self.total_trades += 1
        if profit > 0:
            self.profitable_trades += 1
        self.total_profit += profit

        # Сбрасываем состояние позиции
        self.position_size = 0.0
        self.position_type = None
        self.avg_entry_price = 0.0
        self.entry_step = 0
        self.max_price_since_entry = 0.0
        self.min_price_since_entry = float('inf')
        self.trailing_stop_price = None
        self.trailing_tp_active = False
        self.trailing_tp_price = None
        self.trailing_tp_armed = False
        self.partial_closes_count = 0

        return profit
    
    def open_long_position(self, price: float, position_frac: float) -> bool:
        if self.position_size != 0:
            return False

        # Расчёт размера позиции
        position_usdt = self.config.initial_deposit * position_frac
        position_units = position_usdt / price
        cost_with_fee = position_units * price * (1 + self.config.taker_fee)

        # Минимальный размер позиции
        if position_units <= 1e-8:
            return False

        # Открываем длинную позицию
        self.position_size = float(position_units)
        self.position_type = 'long'
        self.avg_entry_price = float(price)
        self.entry_step = 0
        self.max_price_since_entry = price
        self.min_price_since_entry = price
        self.trailing_tp_armed = False

        self.logger.info(f"Long position opened: {position_frac*100:.1f}% at {price:.2f}")
        return True

    def open_short_position(self, price: float, position_frac: float) -> bool:
        if self.position_size != 0:
            return False

        # Расчёт размера позиции (для шорта используем весь депозит)
        position_usdt = self.config.initial_deposit * position_frac
        position_units = position_usdt / price  # Размер в единицах актива

        # Минимальный размер позиции
        if position_units <= 1e-8:
            return False

        # Открываем короткую позицию
        self.position_size = float(position_units)
        self.position_type = 'short'
        self.avg_entry_price = float(price)
        self.entry_step = 0
        self.max_price_since_entry = price
        self.min_price_since_entry = price
        self.trailing_tp_armed = False

        self.logger.info(f"Short position opened: {position_frac*100:.1f}% at {price:.2f}")
        return True
    
    def update_position(self, current_price: float, atr: float, step: int):
        if self.position_size <= 0:
            return 0.0
        
        self.entry_step = step
        self.max_price_since_entry = max(self.max_price_since_entry, current_price)
        self.min_price_since_entry = min(self.min_price_since_entry, current_price)
        
        # Обрабатываем различные типы закрытия позиций
        results = []

        # 0. Выход по безубытку + профит (3 % или 5 % в зависимости от размера позиции)
        breakeven_profit = self.handle_breakeven_exit(current_price)
        if breakeven_profit is not None:
            results.append(breakeven_profit)

        # 1. Частичный тейк-профит
        partial_profit = self.handle_partial_take_profit(current_price)
        if partial_profit is not None:
            results.append(partial_profit)
        
        # 2. Трейлинг-тейк-профит
        trailing_profit = self.handle_trailing_take_profit(current_price, atr)
        if trailing_profit is not None:
            results.append(trailing_profit)
        
        # 3. Динамический стоп-лосс
        sl_profit = self.handle_dynamic_stop_loss(current_price, atr)
        if sl_profit is not None:
            results.append(sl_profit)
        
        return sum(results) if results else 0.0

    # ------------------------------------------------------------------
    # Новая логика выхода в безубыток + целевой профит
    # ------------------------------------------------------------------
    def handle_breakeven_exit(self, current_price: float) -> Optional[float]:
        """Закрытие всей позиции при достижении небольшого, но значимого профита.

        • Для первой ступени (объём ≈10 % депозита) – цель +3 %
        • Если добавлены 2-я/3-я ступени (объём >15 % депозита) – цель +5 %
        Не выполняется, если частичные ТП уже сработали.
        """
        if self.position_size <= 0:
            return None

        # При срабатывании частичных ТП пропускаем, чтобы не закрывать остаток слишком рано
        if self.partial_closes_count > 0:
            return None

        # Оцениваем долю депозита, вложенную в позицию
        invested_usdt = self.position_size * self.avg_entry_price
        invested_frac = invested_usdt / max(1e-8, self.config.initial_deposit)

        # Целевой процент профита
        target_pct = 0.03 if invested_frac <= 0.15 else 0.05

        # Учитываем комиссию на вход и на выход (дважды taker_fee)
        fee_buffer = 2 * self.config.taker_fee

        target_price = self.avg_entry_price * (1 + target_pct + fee_buffer)

        if current_price >= target_price:
            profit = self._close_position(current_price, 'sell_breakeven_tp')
            self.logger.info(
                f"Breakeven-TP triggered (target {target_pct*100:.1f}%): {profit:.2f} USDT")
            return profit

        return None
    
    def get_position_info(self) -> Dict[str, Any]:
        return {
            'position_size': self.position_size,
            'avg_entry_price': self.avg_entry_price,
            'entry_step': self.entry_step,
            'max_price_since_entry': self.max_price_since_entry,
            'min_price_since_entry': self.min_price_since_entry,
            'trailing_stop_price': self.trailing_stop_price,
            'trailing_tp_active': self.trailing_tp_active,
            'trailing_tp_price': self.trailing_tp_price,
            'trailing_tp_armed': self.trailing_tp_armed,
            'partial_closes_count': self.partial_closes_count,
            'current_volatility': self.current_volatility,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'total_profit': self.total_profit,
            'win_rate': self.profitable_trades / max(1, self.total_trades) * 100
        }
    
    def reset(self):
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.entry_step = 0
        self.max_price_since_entry = 0.0
        self.min_price_since_entry = float('inf')
        self.trailing_stop_price = None
        self.trailing_tp_active = False
        self.trailing_tp_price = None
        self.trailing_tp_armed = False
        self.partial_closes_count = 0
        self.volatility_history.clear()
        self.current_volatility = 1.0
