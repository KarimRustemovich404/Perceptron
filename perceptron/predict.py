"""
Прогноз: загружает модель и веса, предсказывает на датасете и оценивает бинарной кросс-энтропией.

python predict.py --model saved_model.npz --dataset data/data_validation.csv
"""
import argparse

import numpy as np
import pandas as pd

from mlp.losses import binary_cross_entropy
from mlp.model import Network
from mlp.preprocessing import load_dataset, read_raw


def main():
    p = argparse.ArgumentParser(description="Прогноз MLP")
    p.add_argument("--model", default="saved_model.npz")
    p.add_argument("--dataset", default="data/data_validation.csv")
    p.add_argument("--output", help="CSV для сохранения прогнозов (необязательно)")
    args = p.parse_args()

    network, extra = Network.load(args.model)
    X, y = load_dataset(args.dataset)
    X = (X - extra["scaler_mean"]) / extra["scaler_std"]

    P = network.predict_proba(X)
    p_malignant = P[:, 1]
    y_pred = np.argmax(P, axis=1)

    tp = int(np.sum((y_pred == 1) & (y == 1)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"samples              : {len(y)}")
    print(f"binary cross-entropy : {binary_cross_entropy(p_malignant, y):.4f}")
    print(f"accuracy             : {np.mean(y_pred == y):.4f}")
    print(f"precision (M)        : {precision:.4f}")
    print(f"recall (M)           : {recall:.4f}")
    print(f"f1 (M)               : {f1:.4f}")
    print("confusion matrix     :       pred B  pred M")
    print(f"                       true B  {tn:6d}  {fp:6d}")
    print(f"                       true M  {fn:6d}  {tp:6d}")

    if args.output:
        df = read_raw(args.dataset)
        pd.DataFrame({
            "id": df["id"],
            "true": df["diagnosis"],
            "pred": np.where(y_pred == 1, "M", "B"),
            "p_malignant": p_malignant.round(4),
        }).to_csv(args.output, index=False)
        print(f"> прогнозы сохранены в '{args.output}'")


if __name__ == "__main__":
    main()
