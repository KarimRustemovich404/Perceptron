"""
Обучение многослойного персептрона.

Через аргументы:
  python train.py --layer 24 24 24 --epochs 84 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.0314
Через конфиг-файл (имеет приоритет над --layer/--activation/...):
  python train.py --config network.json
"""
import argparse
import json

import matplotlib
import numpy as np

from mlp import DenseLayer, createNetwork, fit
from mlp.preprocessing import StandardScaler, load_dataset, one_hot


def parse_args():
    p = argparse.ArgumentParser(description="Обучение MLP")
    p.add_argument("--dataset_train", default="data/data_training.csv")
    p.add_argument("--dataset_valid", default="data/data_validation.csv")
    p.add_argument("--config", help="JSON-файл с архитектурой и гиперпараметрами")
    p.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="размеры скрытых слоёв")
    p.add_argument("--activation", default="sigmoid", help="активация скрытых слоёв")
    p.add_argument("--weights_initializer", default="heUniform")
    p.add_argument("--epochs", type=int, default=84)
    p.add_argument("--loss", default="binaryCrossentropy")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.0314)
    p.add_argument("--patience", type=int, default=0, help="ранняя остановка (0 — выкл.)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="saved_model.npz")
    p.add_argument("--plot", default="learning_curves.png")
    p.add_argument("--no_show", action="store_true", help="не открывать окно с графиками")
    return p.parse_args()


def build_layers(args, n_features, n_classes):
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for key in ("epochs", "loss", "batch_size", "learning_rate", "patience", "seed"):
            if key in cfg:
                setattr(args, key, cfg[key])
        hidden = cfg["layers"]
        out = cfg.get("output", {})
        layers = [DenseLayer(n_features)]
        layers += [DenseLayer(h["units"], h.get("activation", "sigmoid"),
                              h.get("weights_initializer", "heUniform")) for h in hidden]
        layers.append(DenseLayer(n_classes, "softmax", out.get("weights_initializer", "heUniform")))
        return layers

    layers = [DenseLayer(n_features)]
    layers += [DenseLayer(u, args.activation, args.weights_initializer) for u in args.layer]
    layers.append(DenseLayer(n_classes, "softmax", args.weights_initializer))
    return layers


def plot_history(history, path, show):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history["loss"], label="training loss")
    ax1.plot(epochs, history["val_loss"], "--", label="validation loss")
    ax1.set_xlabel("epochs"); ax1.set_ylabel("loss"); ax1.set_title("Loss")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(epochs, history["acc"], label="training acc")
    ax2.plot(epochs, history["val_acc"], label="validation acc")
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy"); ax2.set_title("Learning Curves")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"> графики сохранены в '{path}'")
    if show:
        plt.show()


def main():
    args = parse_args()

    X_train, y_train = load_dataset(args.dataset_train)
    X_valid, y_valid = load_dataset(args.dataset_valid)

    scaler = StandardScaler().fit(X_train)  # статистики только по train!
    X_train, X_valid = scaler.transform(X_train), scaler.transform(X_valid)
    Y_train, Y_valid = one_hot(y_train), one_hot(y_valid)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    layers = build_layers(args, X_train.shape[1], Y_train.shape[1])
    if len(layers) - 2 < 2:
        print("! внимание: по заданию нужно минимум два скрытых слоя")
    network = createNetwork(layers, seed=args.seed)
    print("архитектура:", " -> ".join(str(l.units) for l in [network.input_layer] + network.layers))

    history = fit(network, (X_train, Y_train), (X_valid, Y_valid), loss=args.loss,
                  learning_rate=args.learning_rate, batch_size=args.batch_size,
                  epochs=args.epochs, patience=args.patience)

    print(f"> saving model '{args.output}' to disk...")
    network.save(args.output, extra={"scaler_mean": scaler.mean, "scaler_std": scaler.std})
    plot_history(history, args.plot, show=not args.no_show)


if __name__ == "__main__":
    np.seterr(over="ignore")
    main()
