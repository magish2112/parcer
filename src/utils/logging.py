#!/usr/bin/env python3
"""
Централизованная система логирования
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from .config import LoggingConfig


def setup_logging(config: LoggingConfig, logger_name: str = "crypto_trading") -> logging.Logger:
    """Настройка централизованного логирования"""
    
    # Создаем директорию для логов
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(config.format)
    
    # Файловый обработчик с ротацией
    log_file = log_dir / config.log_file
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.level.upper()))
    
    # Добавляем обработчики
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Предотвращаем дублирование логов
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Получение логгера по имени"""
    return logging.getLogger(f"crypto_trading.{name}")


class TradingLogger:
    """Специализированный логгер для торговых операций"""
    
    def __init__(self, config: LoggingConfig):
        self.logger = setup_logging(config, "trading")
        self.trade_logger = get_logger("trades")
        self.error_logger = get_logger("errors")
        self.performance_logger = get_logger("performance")
    
    def log_trade(self, message: str, **kwargs):
        """Логирование торговых операций"""
        self.trade_logger.info(f"[TRADE] {message}", extra=kwargs)
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Логирование ошибок"""
        if exception:
            self.error_logger.error(f"[ERROR] {message}: {exception}", exc_info=True, extra=kwargs)
        else:
            self.error_logger.error(f"[ERROR] {message}", extra=kwargs)
    
    def log_performance(self, message: str, **kwargs):
        """Логирование производительности"""
        self.performance_logger.info(f"[PERF] {message}", extra=kwargs)
    
    def log_info(self, message: str, **kwargs):
        """Общее информационное логирование"""
        self.logger.info(f"[INFO] {message}", extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Логирование предупреждений"""
        self.logger.warning(f"[WARN] {message}", extra=kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Отладочное логирование"""
        self.logger.debug(f"[DEBUG] {message}", extra=kwargs)
