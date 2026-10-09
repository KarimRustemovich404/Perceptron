"""
Функции потерь. forward возвращает число, backward — dL/dY_pred (с учётом усреднения по батчу).

Бинарная кросс-энтропия (p — вероятность класса 1):
    E = -1/N * sum_n [ y_n log p_n + (1 - y_n) log(1 - p_n) ]

При softmax-выходе из 2 нейронов и one-hot метках это та же формула:
    E = -1/N * sum_n sum_k y_nk log p_nk,
так как p_n0 = 1 - p_n1 и y_n0 = 1 - y_n1.
"""
import numpy as np

EPS = 1e-15


class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        p = np.clip(y_pred, EPS, 1 - EPS)
        if p.shape[1] == 1:  # один сигмоидный выход
            return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    def backward(self, y_pred, y_true):
        p = np.clip(y_pred, EPS, 1 - EPS)
        n = y_true.shape[0]
        if p.shape[1] == 1:
            return -(y_true / p - (1 - y_true) / (1 - p)) / n
        return -(y_true / p) / n


class MeanSquaredError:
    def forward(self, y_pred, y_true):
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]


LOSSES = {
    "binaryCrossentropy": BinaryCrossEntropy,
    "categoricalCrossentropy": BinaryCrossEntropy,  # для 2 классов совпадает
    "mse": MeanSquaredError,
}


def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Неизвестная функция потерь '{name}'. Доступны: {list(LOSSES)}")
    return LOSSES[name]()


def binary_cross_entropy(p, y):
    """BCE по вероятности класса 1 (p) и меткам 0/1 (y) — для оценки в predict.py."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
