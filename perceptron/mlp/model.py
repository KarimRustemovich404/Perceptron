"""
Сеть = последовательность Dense-слоёв.

Обучение (одна итерация на мини-батче):
  1. Прямой проход: A0 = X, A_l = f_l(A_{l-1} W_l + b_l), выход Y_hat = A_L.
  2. Потеря L(Y_hat, Y).
  3. Обратный проход: от dL/dY_hat к первому слою, каждый слой считает dW, db.
  4. Градиентный спуск: W <- W - lr * dW, b <- b - lr * db.
"""
import json

import numpy as np

from .layers import DenseLayer
from .losses import get_loss


class Network:
    def __init__(self, layers, seed=42):
        """
        layers[0] — входной слой: задаёт только размерность входа (units = число признаков),
        весов не имеет. Остальные слои — скрытые и выходной.
        """
        if len(layers) < 2:
            raise ValueError("Нужны как минимум входной и выходной слои")
        self.input_layer = layers[0]
        self.layers = layers[1:]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        fan_in = self.input_layer.units
        for layer in self.layers:
            layer.build(fan_in, self.rng)
            fan_in = layer.units

    # ---------- прямой / обратный проход ----------
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    # ---------- прогноз ----------
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    # ---------- обучение ----------
    def fit(self, X_train, Y_train, X_valid, Y_valid, loss="binaryCrossentropy",
            learning_rate=0.01, batch_size=32, epochs=100, patience=0, verbose=True):
        loss_fn = get_loss(loss)
        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        n = X_train.shape[0]
        best_val, best_weights, wait = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            idx = self.rng.permutation(n)  # перемешивание каждую эпоху
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                Y_hat = self.forward(X_train[b])
                self.backward(loss_fn.backward(Y_hat, Y_train[b]))
                self.step(learning_rate)

            # метрики по всей выборке в конце эпохи
            P_train, P_valid = self.forward(X_train), self.forward(X_valid)
            history["loss"].append(loss_fn.forward(P_train, Y_train))
            history["val_loss"].append(loss_fn.forward(P_valid, Y_valid))
            history["acc"].append(accuracy(P_train, Y_train))
            history["val_acc"].append(accuracy(P_valid, Y_valid))

            if verbose:
                w = len(str(epochs))
                print(f"epoch {epoch:0{w}d}/{epochs} - loss: {history['loss'][-1]:.4f} "
                      f"- val_loss: {history['val_loss'][-1]:.4f} "
                      f"- acc: {history['acc'][-1]:.4f} - val_acc: {history['val_acc'][-1]:.4f}")

            # ранняя остановка (бонус): patience=0 — выключена
            if patience > 0:
                if history["val_loss"][-1] < best_val - 1e-6:
                    best_val, wait = history["val_loss"][-1], 0
                    best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"> early stopping: val_loss не улучшался {patience} эпох, "
                                  f"возвращаем лучшие веса (val_loss={best_val:.4f})")
                        for l, (W, bb) in zip(self.layers, best_weights):
                            l.W, l.b = W, bb
                        break
        return history

    # ---------- сохранение / загрузка ----------
    def config(self):
        return {
            "seed": self.seed,
            "layers": [{"units": self.input_layer.units, "activation": "input"}]
                      + [l.config() for l in self.layers],
        }

    def save(self, path, extra=None):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        for k, v in (extra or {}).items():
            arrays[k] = v
        arrays["config"] = np.array(json.dumps(self.config()))
        np.savez(path, **arrays)

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)
        cfg = json.loads(str(data["config"]))
        specs = cfg["layers"]
        layers = [DenseLayer(specs[0]["units"])]
        layers += [DenseLayer(s["units"], s["activation"], s["weights_initializer"]) for s in specs[1:]]
        net = Network(layers, seed=cfg["seed"])
        for i, layer in enumerate(net.layers):
            layer.W, layer.b = data[f"W{i}"], data[f"b{i}"]
        extra = {k: data[k] for k in data.files if not (k == "config" or k[0] in "Wb" and k[1:].isdigit())}
        return net, extra


def accuracy(P, Y):
    return float(np.mean(np.argmax(P, axis=1) == np.argmax(Y, axis=1)))


# ---------- API в стиле примера из задания ----------
def createNetwork(layers, seed=42):
    return Network(layers, seed=seed)


def fit(network, data_train, data_valid, loss="binaryCrossentropy", learning_rate=0.01,
        batch_size=32, epochs=100, patience=0, verbose=True):
    (X_tr, Y_tr), (X_va, Y_va) = data_train, data_valid
    return network.fit(X_tr, Y_tr, X_va, Y_va, loss=loss, learning_rate=learning_rate,
                       batch_size=batch_size, epochs=epochs, patience=patience, verbose=verbose)
