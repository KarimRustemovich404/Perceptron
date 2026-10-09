"""
Разделение data.csv на обучающую и валидационную выборки.
Стратифицированное: доля M/B одинакова в обеих частях. Фиксированный seed -> воспроизводимость.

python split.py --dataset data.csv --valid_ratio 0.2 --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd


def stratified_split(labels, valid_ratio, rng):
    train_idx, valid_idx = [], []
    for cls in np.unique(labels):
        idx = rng.permutation(np.where(labels == cls)[0])
        n_valid = int(round(len(idx) * valid_ratio))
        valid_idx.extend(idx[:n_valid])
        train_idx.extend(idx[n_valid:])
    return rng.permutation(train_idx), rng.permutation(valid_idx)


def main():
    p = argparse.ArgumentParser(description="Разделение датасета на train/valid")
    p.add_argument("--dataset", default="data.csv")
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    df = pd.read_csv(args.dataset, header=None)
    rng = np.random.default_rng(args.seed)
    tr, va = stratified_split(df[1].to_numpy(), args.valid_ratio, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "data_training.csv")
    valid_path = os.path.join(args.out_dir, "data_validation.csv")
    df.iloc[tr].to_csv(train_path, header=False, index=False)
    df.iloc[va].to_csv(valid_path, header=False, index=False)

    for name, part in (("train", df.iloc[tr]), ("valid", df.iloc[va])):
        counts = part[1].value_counts()
        print(f"{name}: {len(part)} строк (M={counts.get('M', 0)}, B={counts.get('B', 0)})")
    print(f"> сохранено: {train_path}, {valid_path}")


if __name__ == "__main__":
    main()
