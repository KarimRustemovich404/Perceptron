"""
Инициализация весов. Матрица W имеет форму (fan_in, fan_out).

He   — для ReLU-подобных активаций: дисперсия 2 / fan_in.
Xavier (Glorot) — для sigmoid/tanh: дисперсия 2 / (fan_in + fan_out).
Для равномерного распределения U(-L, L) дисперсия = L^2 / 3, отсюда пределы L.
"""
import numpy as np


def he_uniform(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def xavier_uniform(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def xavier_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def random_small(fan_in, fan_out, rng):
    return rng.normal(0.0, 0.01, size=(fan_in, fan_out))


INITIALIZERS = {
    "heUniform": he_uniform,
    "heNormal": he_normal,
    "xavierUniform": xavier_uniform,
    "xavierNormal": xavier_normal,
    "random": random_small,
}


def get_initializer(name):
    if name not in INITIALIZERS:
        raise ValueError(f"Неизвестный инициализатор '{name}'. Доступны: {list(INITIALIZERS)}")
    return INITIALIZERS[name]
