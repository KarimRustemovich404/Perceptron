"""Загрузка данных и предобработка."""
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
# 30 признаков: среднее, стандартная ошибка и «худшее» (среднее трёх наибольших) значение
FEATURES = ([f"{f}_mean" for f in BASE_FEATURES]
            + [f"{f}_se" for f in BASE_FEATURES]
            + [f"{f}_worst" for f in BASE_FEATURES])
COLUMNS = ["id", "diagnosis"] + FEATURES


def read_raw(path):
    """Читает CSV без заголовка в DataFrame с именованными столбцами."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось {len(COLUMNS)} столбцов, получено {df.shape[1]}")
    df.columns = COLUMNS
    return df


def load_dataset(path):
    """Возвращает X (N x 30) и y (N,) где M -> 1, B -> 0."""
    df = read_raw(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    return X, y


def one_hot(y, n_classes=2):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


class StandardScaler:
    """
    z = (x - mean) / std. Параметры считаются ТОЛЬКО по обучающей выборке и
    сохраняются вместе с моделью, чтобы применить то же преобразование при прогнозе.
    """
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)
