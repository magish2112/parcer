from config import DATASET_PATH
import pandas as pd
import ta
import numpy as np
import os

def get_ohlcv(client=None, symbol=None, interval='240', limit=200):
	"""
	Получить исторические OHLCV-данные — локально из подготовленного CSV.
	При сохранении датасета используйте config.DATASET_PATH.
	"""
	# пытаемся прочитать путь из конфига, если нет — пробуем известные имена
	path = DATASET_PATH if 'DATASET_PATH' in globals() else None
	if not path or not os.path.exists(path):
		# fallback варианты
		for p in ('btc_4h_full.csv', 'btc_4h_ohlcv.csv', 'btc_1h_full.csv'):
			if os.path.exists(p):
				path = p
				break
	if not path or not os.path.exists(path):
		raise FileNotFoundError("OHLCV файл не найден. Проверьте DATASET_PATH или наличие btc_4h_full.csv.")
	df = pd.read_csv(path)
	return df

def add_indicators(df):
	"""
	Добавить расширенные технические индикаторы к DataFrame
	"""
	# Базовые индикаторы
	df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
	df['ema_14'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
	df['macd'] = ta.trend.MACD(df['close']).macd()
	bb = ta.volatility.BollingerBands(df['close'])
	df['bb_bbm'] = bb.bollinger_mavg()
	df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
	df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
	# Новые индикаторы
	df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
	df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
	df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
	# Connors RSI (crsi)
	rsi3 = ta.momentum.RSIIndicator(df['close'], window=3).rsi()
	streak = (np.sign(df['close'].diff()).groupby((np.sign(df['close'].diff()) != np.sign(df['close'].diff().shift())).cumsum()).cumcount() + 1) * np.sign(df['close'].diff())
	streak_rsi = ta.momentum.RSIIndicator(streak.fillna(0), window=2).rsi()
	roc100 = df['close'].pct_change(periods=100) * 100
	roc_rsi = ta.momentum.RSIIndicator(roc100.fillna(0), window=3).rsi()
	df['crsi'] = (rsi3 + streak_rsi + roc_rsi) / 3
	# Боковые объёмы (sideways_volume): средний объём при ADX < 25
	df['sideways_volume'] = df['volume'].rolling(window=20).mean() * (df['adx'] < 25)
	# --- Новые признаки для торговли по уровням ---
	# Уровни поддержки/сопротивления (по rolling min/max)
	df['support'] = df['low'].rolling(window=50).min()
	df['resistance'] = df['high'].rolling(window=50).max()
	df['dist_to_support'] = df['close'] - df['support']
	df['dist_to_resistance'] = df['resistance'] - df['close']
	# Ложный пробой: close вышла за уровень и вернулась
	df['false_breakout'] = (
		((df['close'].shift(1) < df['support'].shift(1)) & (df['close'] > df['support'])) |
		((df['close'].shift(1) > df['resistance'].shift(1)) & (df['close'] < df['resistance']))
	).astype(int)
	# Всплеск объёма
	df['volume_spike'] = (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5).astype(int)
	# Тренд по EMA
	df['trend_ema'] = (df['close'] > df['ema_14']).astype(int)
	
	# Добавляем недостающие признаки для совместимости с env_crypto.py
	df['ema200_d1'] = df['close']  # Заглушка для дневной EMA200
	df['atr_mult_trailing_stop'] = df['atr'] * 1.5  # ATR множитель для трейлинг-стопа
	
	df = df.dropna()
	return df