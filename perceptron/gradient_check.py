"""
Проверка обратного распространения численным градиентом:
    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 eps)
Если backprop реализован верно, относительная ошибка ~1e-7 и меньше.

python gradient_check.py
"""
import numpy as np

from mlp import DenseLayer, createNetwork
from mlp.losses import get_loss


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    Y = np.eye(2)[rng.integers(0, 2, size=5)]
    for act in ("sigmoid", "tanh", "relu"):
        net = createNetwork([DenseLayer(4), DenseLayer(6, act), DenseLayer(5, act),
                             DenseLayer(2, "softmax")], seed=1)
        loss = get_loss("binaryCrossentropy")
        net.backward(loss.backward(net.forward(X), Y))

        worst, eps = 0.0, 1e-6
        for layer in net.layers:
            for param, grad in ((layer.W, layer.dW), (layer.b, layer.db)):
                for i in np.ndindex(param.shape):
                    old = param[i]
                    param[i] = old + eps; lp = loss.forward(net.forward(X), Y)
                    param[i] = old - eps; lm = loss.forward(net.forward(X), Y)
                    param[i] = old
                    num = (lp - lm) / (2 * eps)
                    rel = abs(num - grad[i]) / max(1e-12, abs(num) + abs(grad[i]))
                    worst = max(worst, rel)
        print(f"{act:8s}: максимальная относительная ошибка = {worst:.2e}")


if __name__ == "__main__":
    main()
