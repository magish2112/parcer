#!/usr/bin/env python3
"""
Система мониторинга торгового бота
"""

import time
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .logging import get_logger
from .config import MonitoringConfig


@dataclass
class TradingMetrics:
    """Метрики торговли"""
    timestamp: float
    step: int
    total_trades: int = 0
    profitable_trades: int = 0
    total_profit: float = 0.0
    current_balance: float = 0.0
    position_size: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    equity_peak: float = 0.0
    loss_streak: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    timestamp: float
    step: int
    prediction_time: float = 0.0
    step_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return asdict(self)


class TradingMonitor:
    """Монитор торговых операций"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = get_logger("monitor")
        
        # Метрики
        self.trading_metrics: List[TradingMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Состояние
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.alert_thresholds = config.alert_thresholds
        
        # Создаем директорию для метрик
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def reset_metrics(self):
        """Сброс всех накопленных метрик"""
        self.trading_metrics = []
        self.performance_metrics = []
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.logger.info("Metrics reset - starting fresh")
    
    def update_trading_metrics(self, step: int, info: Dict[str, Any], 
                             trade_log: List[Dict] = None) -> TradingMetrics:
        """Обновление торговых метрик"""
        try:
            # Базовые метрики
            total_trades = len(info.get('trades', []))
            profitable_trades = sum(1 for trade in info.get('trades', []) if trade > 0)
            total_profit = sum(info.get('trades', []))
            current_balance = info.get('balance', 0.0)
            position_size = info.get('position_size', 0.0)
            equity_peak = info.get('equity_peak', current_balance)
            current_drawdown = info.get('drawdown', 0.0)
            
            # Расчетные метрики (только из текущей сессии)
            # Используем только текущие сделки для алертов
            current_trades = info.get('trades', [])
            current_profitable = sum(1 for trade in current_trades if trade > 0)
            win_rate = (current_profitable / len(current_trades) * 100) if current_trades else 0.0
            avg_profit_per_trade = (total_profit / total_trades) if total_trades > 0 else 0.0
            
            # Максимальная просадка (только из текущей сессии)
            # Используем только текущую просадку для алертов
            max_drawdown = current_drawdown
            
            # Sharpe ratio (упрощенный)
            trades = info.get('trades', [])
            if len(trades) > 1:
                returns = np.array(trades)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Лучшие и худшие сделки
            best_trade = max(trades) if trades else 0.0
            worst_trade = min(trades) if trades else 0.0
            
            # Серия убытков
            loss_streak = 0
            for trade in reversed(trades):
                if trade < 0:
                    loss_streak += 1
                else:
                    break
            
            # Создаем метрики
            metrics = TradingMetrics(
                timestamp=time.time(),
                step=step,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                total_profit=total_profit,
                current_balance=current_balance,
                position_size=position_size,
                win_rate=win_rate,
                avg_profit_per_trade=avg_profit_per_trade,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                equity_peak=equity_peak,
                loss_streak=loss_streak,
                best_trade=best_trade,
                worst_trade=worst_trade
            )
            
            # Добавляем в список
            self.trading_metrics.append(metrics)
            
            # Проверяем алерты
            self._check_alerts(metrics)
            
            # Сохраняем метрики если нужно
            if step % self.config.save_interval == 0:
                self._save_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating trading metrics: {e}")
            return TradingMetrics(timestamp=time.time(), step=step)
    
    def update_performance_metrics(self, step: int, prediction_time: float = 0.0, 
                                 step_time: float = 0.0) -> PerformanceMetrics:
        """Обновление метрик производительности"""
        try:
            # Получаем использование ресурсов
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                step=step,
                prediction_time=prediction_time,
                step_time=step_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            )
            
            self.performance_metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
            return PerformanceMetrics(timestamp=time.time(), step=step)
    
    def _check_alerts(self, metrics: TradingMetrics):
        """Проверка алертов (отключено для чистого обучения)"""
        try:
            # Временно отключаем алерты для чистого обучения
            # Проверяем только критические ошибки
            if metrics.max_drawdown > 0.95:  # Только при просадке > 95%
                self.logger.warning(f"CRITICAL: Extreme drawdown {metrics.max_drawdown:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _get_memory_usage(self) -> float:
        """Получение использования памяти"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Получение использования CPU"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def _save_metrics(self):
        """Сохранение метрик"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Сохраняем торговые метрики
            if self.trading_metrics:
                trading_file = self.metrics_dir / f"trading_metrics_{timestamp}.json"
                with open(trading_file, 'w') as f:
                    json.dump([m.to_dict() for m in self.trading_metrics], f, indent=2)
            
            # Сохраняем метрики производительности
            if self.performance_metrics:
                perf_file = self.metrics_dir / f"performance_metrics_{timestamp}.json"
                with open(perf_file, 'w') as f:
                    json.dump([m.to_dict() for m in self.performance_metrics], f, indent=2)
            
            self.logger.info(f"Metrics saved to {self.metrics_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик"""
        try:
            if not self.trading_metrics:
                return {'status': 'no_data'}
            
            latest = self.trading_metrics[-1]
            
            return {
                'status': 'active',
                'total_steps': latest.step,
                'total_trades': latest.total_trades,
                'win_rate': latest.win_rate,
                'total_profit': latest.total_profit,
                'current_balance': latest.current_balance,
                'max_drawdown': latest.max_drawdown,
                'current_drawdown': latest.current_drawdown,
                'sharpe_ratio': latest.sharpe_ratio,
                'loss_streak': latest.loss_streak,
                'best_trade': latest.best_trade,
                'worst_trade': latest.worst_trade,
                'uptime': time.time() - self.start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def export_to_csv(self, filepath: str = None):
        """Экспорт метрик в CSV"""
        try:
            if not self.trading_metrics:
                self.logger.warning("No trading metrics to export")
                return
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.metrics_dir / f"trading_metrics_{timestamp}.csv"
            
            # Создаем DataFrame
            df = pd.DataFrame([m.to_dict() for m in self.trading_metrics])
            
            # Сохраняем
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def reset(self):
        """Сброс монитора"""
        self.trading_metrics.clear()
        self.performance_metrics.clear()
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.logger.info("Monitor reset")
