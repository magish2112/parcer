#!/usr/bin/env python3
"""
Скрипт для мониторинга прогресса обучения
"""

import time
import pandas as pd
from pathlib import Path

def monitor_training():
    """Мониторинг прогресса обучения"""
    print("🔍 Мониторинг обучения... (Ctrl+C для выхода)")

    last_timesteps = 0
    start_time = time.time()

    try:
        while True:
            # Проверяем файл прогресса
            progress_file = Path("logs/progress.csv")
            if progress_file.exists():
                try:
                    # Читаем последнюю строку
                    with open(progress_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Пропускаем заголовок
                            last_line = lines[-1].strip()
                            parts = last_line.split(',')

                            if len(parts) >= 2:
                                timesteps = int(float(parts[0]))
                                elapsed_time = float(parts[1])

                                if timesteps > last_timesteps:
                                    progress_pct = (timesteps / 1000000) * 100
                                    fps = timesteps / elapsed_time if elapsed_time > 0 else 0

                                    print(f"Шаги: {timesteps:>6d} | "
                                          f"Прогресс: {progress_pct:>5.1f}% | "
                                          f"FPS: {fps:>3.0f} | "
                                          f"Время: {elapsed_time:>4.1f} сек")

                                    # Показываем дополнительные метрики если доступны
                                    if len(parts) > 5:
                                        try:
                                            reward_mean = float(parts[5])
                                            ep_len_mean = float(parts[4]) if len(parts) > 4 else 0
                                            print(f"Средняя награда: {reward_mean:>.2f} | Средняя длина эпизода: {ep_len_mean:.0f}")
                                        except
                                            pass

                                    last_timesteps = timesteps

                except Exception as e:
                    print(f"Ошибка чтения прогресса: {e}")

            time.sleep(30)  # Проверяем каждые 30 секунд

    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print("\n⏹️  Мониторинг остановлен")
        print(f"Время работы: {total_time:.1f} сек")
        if last_timesteps > 0:
            print(f"Последний прогресс: {last_timesteps} шагов")
if __name__ == "__main__":
    monitor_training()
