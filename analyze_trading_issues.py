#!/usr/bin/env python3
"""
Анализ проблем с микросделками и стоп-лоссами
"""

import pandas as pd
import numpy as np

def analyze_trading_issues():
    """Анализ проблем в торговой логике"""
    
    # Анализ логов обучения
    df = pd.read_csv('adaptive_training_trades.csv')
    
    print('🔍 АНАЛИЗ ПРОБЛЕМ С МИКРОСДЕЛКАМИ И СТОП-ЛОССАМИ')
    print('=' * 60)
    
    # Анализ размеров прибыли
    profits = df['profit']
    print(f'📊 Статистика прибыли:')
    print(f'   - Средняя прибыль: {profits.mean():.2f} USDT')
    print(f'   - Медианная прибыль: {profits.median():.2f} USDT')
    print(f'   - Стандартное отклонение: {profits.std():.2f} USDT')
    
    # Анализ микросделок
    micro_trades = df[abs(profits) < 10]  # Сделки менее 10 USDT
    print(f'\n⚠️  Микросделки (< 10 USDT):')
    print(f'   - Количество: {len(micro_trades)} ({len(micro_trades)/len(df)*100:.1f}%)')
    print(f'   - Средняя прибыль: {micro_trades["profit"].mean():.2f} USDT')
    print(f'   - Общая прибыль: {micro_trades["profit"].sum():.2f} USDT')
    
    # Анализ по типам действий
    print(f'\n📈 Анализ по типам действий:')
    action_stats = df.groupby('action')['profit'].agg(['count', 'mean', 'sum'])
    for action, stats in action_stats.iterrows():
        print(f'   {action}:')
        print(f'     - Количество: {stats["count"]} ({stats["count"]/len(df)*100:.1f}%)')
        print(f'     - Средняя прибыль: {stats["mean"]:.2f} USDT')
        print(f'     - Общая прибыль: {stats["sum"]:.2f} USDT')
    
    # Анализ частоты стоп-лоссов
    stop_losses = df[df['action'] == 'sell_stop_loss']
    print(f'\n🛑 Анализ стоп-лоссов:')
    print(f'   - Количество: {len(stop_losses)} ({len(stop_losses)/len(df)*100:.1f}%)')
    print(f'     - Средний убыток: {stop_losses["profit"].mean():.2f} USDT')
    print(f'     - Общий убыток: {stop_losses["profit"].sum():.2f} USDT')
    
    # Анализ трейлинг-тейк-профитов
    trailing_tp = df[df['action'] == 'sell_trailing_tp']
    print(f'\n📈 Анализ трейлинг-тейк-профитов:')
    print(f'   - Количество: {len(trailing_tp)} ({len(trailing_tp)/len(df)*100:.1f}%)')
    print(f'     - Средняя прибыль: {trailing_tp["profit"].mean():.2f} USDT')
    print(f'     - Общая прибыль: {trailing_tp["profit"].sum():.2f} USDT')
    
    # Анализ частичных тейк-профитов
    partial_tp = df[df['action'] == 'partial_tp']
    print(f'\n💰 Анализ частичных тейк-профитов:')
    print(f'   - Количество: {len(partial_tp)} ({len(partial_tp)/len(df)*100:.1f}%)')
    if len(partial_tp) > 0:
        print(f'     - Средняя прибыль: {partial_tp["profit"].mean():.2f} USDT')
        print(f'     - Общая прибыль: {partial_tp["profit"].sum():.2f} USDT')
    else:
        print('     - Частичные тейк-профиты не использовались')
    
    # Анализ комиссий
    total_fees = abs(df['profit'] * 0.001).sum()  # Примерная оценка комиссий
    print(f'\n💸 Оценка комиссий:')
    print(f'   - Общие комиссии: ~{total_fees:.2f} USDT')
    print(f'   - Комиссии как % от оборота: {total_fees/abs(df["profit"]).sum()*100:.2f}%')
    
    # Анализ проблем
    print(f'\n🚨 ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:')
    
    # Проблема 1: Слишком много микросделок
    micro_ratio = len(micro_trades) / len(df)
    if micro_ratio > 0.3:
        print(f'   1. ❌ Слишком много микросделок: {micro_ratio*100:.1f}%')
        print(f'      Рекомендация: Ввести минимальный порог прибыли')
    
    # Проблема 2: Частые стоп-лоссы
    sl_ratio = len(stop_losses) / len(df)
    if sl_ratio > 0.4:
        print(f'   2. ❌ Слишком частые стоп-лоссы: {sl_ratio*100:.1f}%')
        print(f'      Рекомендация: Увеличить trailing_stop_multiplier')
    
    # Проблема 3: Низкая эффективность трейлинг-тейк-профитов
    if len(trailing_tp) > 0:
        tp_avg_profit = trailing_tp["profit"].mean()
        if tp_avg_profit < 20:
            print(f'   3. ❌ Низкая эффективность трейлинг-ТП: {tp_avg_profit:.2f} USDT')
            print(f'      Рекомендация: Увеличить trailing_tp_trailing_pct')
    
    # Проблема 4: Отсутствие частичных тейк-профитов
    if len(partial_tp) == 0:
        print(f'   4. ❌ Частичные тейк-профиты не используются')
        print(f'      Рекомендация: Активировать partial_tp_pct')
    
    print(f'\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:')
    print(f'   1. Ввести минимальный порог для частичных фиксаций')
    print(f'   2. Добавить "arm"-порог для трейлинга (активировать только после покрытия комиссий)')
    print(f'   3. Рассмотреть менее "тесный" трейлинг по локальной структуре')
    print(f'   4. Увеличить trailing_stop_multiplier для сокращения частых стоп-лоссов')
    print(f'   5. Добавить анализ волатильности для адаптивных параметров')

if __name__ == '__main__':
    analyze_trading_issues()
