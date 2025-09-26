from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import os
import warnings
try:
	from sklearn.exceptions import InconsistentVersionWarning
	warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
	# sklearn версий может не содержать этот класс в старых версиях — безопасно игнорируем
	pass

MODEL_PATH = "ppo_crypto_model.zip"
VEC_PATH = "vecnormalize.pkl"

def _ensure_reset_returns_pair(env):
    """
    Если env.reset() возвращает только observation, переопределяем его
    чтобы всегда возвращать (observation, info_dict).
    Это меняет метод на самом объекте env (монкипатч).
    """
    if not hasattr(env, 'reset'):
        return
    orig_reset = env.reset
    def wrapped_reset(*args, **kwargs):
        res = orig_reset(*args, **kwargs)
        # Если уже кортеж длины 2 — возвращаем как есть
        if isinstance(res, tuple) and len(res) == 2:
            return res
        # Иначе возвращаем (obs, {})
        return res, {}
    # Заменяем метод
    try:
        env.reset = wrapped_reset
    except Exception:
        # в редких случаях объект может быть иммутабельным — тогда пропускаем
        pass

def _adapt_observation_to_model(model, state: np.ndarray) -> np.ndarray:
    """
    Привести state к форме, ожидаемой моделью.
    - state: np.ndarray формы (n_env, w, f) или (w, f) или (1, w, f)
    - возвращает np.ndarray формы (n_env, exp_w, exp_f)
    """
    # Получаем ожидаемую форму (w, f)
    expected = None
    try:
        exp_shape = model.policy.observation_space.shape
    except Exception:
        exp_shape = getattr(getattr(model, 'observation_space', None), 'shape', None)
    if exp_shape is None:
        return state  # ничего не знаем — возвращаем как есть

    # нормализация: exp_shape может быть (w, f) или (n_env, w, f)
    if len(exp_shape) == 3:
        exp_w, exp_f = exp_shape[1], exp_shape[2]
    elif len(exp_shape) == 2:
        exp_w, exp_f = exp_shape
    else:
        return state

    arr = np.array(state)
    # привести к форме (n_env, w, f)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)  # (1, w, f)
    if arr.ndim == 3:
        n_env, w, f = arr.shape
    else:
        # непредвиденная форма
        return arr

    # если количество фич не совпадает — не пытаемся гадать
    if f != exp_f:
        # попытка автоматической коррекции невозможна — вернуть оригинал и позволить SB3 выбросить понятную ошибку
        return arr

    # если текущее окно меньше — pad сверху повторением первой строки (консервативно)
    if w < exp_w:
        pad_top = exp_w - w
        first_row = arr[:, 0:1, :]  # (n_env,1,f)
        pad = np.repeat(first_row, pad_top, axis=1)  # (n_env,pad_top,f)
        arr = np.concatenate([pad, arr], axis=1)
    # если текущее окно больше — возьмём последние exp_w шагов
    elif w > exp_w:
        arr = arr[:, -exp_w:, :]

    # теперь arr.shape == (n_env, exp_w, exp_f)
    return arr

def load_or_create_model(env):
    """
    Загружает обученную модель PPO и (опционально) VecNormalize, или создаёт новую.
    """
    # если есть сохранённый нормализатор, подгружаем его в переданную env
    if os.path.exists(VEC_PATH):
        try:
            env = VecNormalize.load(VEC_PATH, env)
        except Exception:
            # если не удалось загрузить нормализатор — продолжаем с переданной средой
            pass

    # Применяем обёртку reset чтобы гарантировать (obs, info) интерфейс
    _ensure_reset_returns_pair(env)

    if os.path.exists(MODEL_PATH):
        try:
            # пробуем загрузить модель с переданной средой (SB3 проверяет spaces)
            model = PPO.load(MODEL_PATH, env=env)
            # показать, какую форму наблюдения ожидает модель (полезно для отладки window_size)
            try:
                print(f"Loaded model expects observation shape: {model.policy.observation_space.shape}")
            except Exception:
                pass
        except ValueError as e:
            # несовпадение пространств (например, разный window_size)
            # fallback: загрузим модель без привязки к env для инференса
            print(f"Warning: cannot load model with provided env (spaces mismatch): {e}")
            print("Loading model without env (fallback). Use compatible env for training/eval if needed.")
            model = PPO.load(MODEL_PATH)
            try:
                print(f"Loaded model (fallback) expects observation shape: {model.policy.observation_space.shape}")
            except Exception:
                pass
    else:
        # создаём новую модель и сохраняем вместе с нормализатором (если он есть)
        model = PPO('MlpPolicy', env, verbose=0)
        model.save(MODEL_PATH)
        try:
            # если env — VecNormalize, сохраняем его статистику
            if hasattr(env, 'save'):
                env.save(VEC_PATH)
        except Exception:
            pass
    return model

def get_action(model, state):
    """
    Получить торговый сигнал от модели (скаляр int).
    Адаптируем state под ожидаемую модель, если window_size не совпадает.
    """
    # Приводим state к numpy
    state_arr = np.array(state)
    # Подгоняем форму под модель (padding/crop по времени)
    try:
        state_arr = _adapt_observation_to_model(model, state_arr)
    except Exception:
        # в крайнем случае используем оригинал
        state_arr = np.array(state)

    # SB3 ожидает либо (obs,) соответствующее observation_space — подаём state_arr напрямую
    action, _ = model.predict(state_arr, deterministic=True)
    # action может быть массивом/скаляром — приводим к int
    if isinstance(action, (list, tuple, np.ndarray)):
        return int(action[0])
    return int(action)
