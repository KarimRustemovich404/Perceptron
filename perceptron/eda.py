"""
Предварительный анализ данных: статистика, баланс классов, связь признаков с диагнозом,
корреляции между признаками. Графики сохраняются в папку eda/.

python eda.py --dataset data.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlp.preprocessing import BASE_FEATURES, FEATURES, read_raw


def main():
    p = argparse.ArgumentParser(description="EDA датасета")
    p.add_argument("--dataset", default="data.csv")
    p.add_argument("--out_dir", default="eda")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 200)

    df = read_raw(args.dataset)
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов (id, diagnosis, 30 признаков)")
    print(f"Пропуски: {int(df.isna().sum().sum())}, дубликаты id: {int(df['id'].duplicated().sum())}")

    counts = df["diagnosis"].value_counts()
    print("\nБаланс классов:")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v / len(df):.1%})")

    print("\nОписательная статистика (*_mean признаки):")
    print(df[[f"{f}_mean" for f in BASE_FEATURES]].describe().T.round(3))

    # масштабы признаков сильно различаются -> нужна стандартизация
    stds = df[FEATURES].std()
    print(f"\nРазброс стандартных отклонений признаков: от {stds.min():.5f} ({stds.idxmin()}) "
          f"до {stds.max():.1f} ({stds.idxmax()}) — нужна нормализация")

    y = (df["diagnosis"] == "M").astype(int)
    corr_target = df[FEATURES].corrwith(y).sort_values(key=np.abs, ascending=False)
    print("\nКорреляция признаков с диагнозом (M=1), топ-10:")
    print(corr_target.head(10).round(3).to_string())
    print("Самые слабые:")
    print(corr_target.tail(5).round(3).to_string())

    corr = df[FEATURES].corr()
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .sort_values(key=np.abs, ascending=False))
    print("\nСильно коррелирующие пары признаков (|r| > 0.95):")
    print(pairs[pairs.abs() > 0.95].round(3).to_string())

    # --- графики ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(counts.index, counts.values, color=["tab:green", "tab:red"])
    ax.set_title("Баланс классов")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "class_balance.png")); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, f in zip(axes.ravel(), BASE_FEATURES):
        col = f"{f}_mean"
        ax.hist(df.loc[df.diagnosis == "B", col], bins=30, alpha=0.6, label="B", color="tab:green")
        ax.hist(df.loc[df.diagnosis == "M", col], bins=30, alpha=0.6, label="M", color="tab:red")
        ax.set_title(col); ax.legend()
    fig.suptitle("Распределения *_mean признаков по классам")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "histograms.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=7)
    fig.colorbar(im); ax.set_title("Корреляционная матрица признаков")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "correlation.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 9))
    ct = corr_target.sort_values()
    ax.barh(ct.index, ct.values, color=["tab:red" if v > 0 else "tab:blue" for v in ct.values])
    ax.set_title("Корреляция признаков с диагнозом (M=1)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "target_correlation.png")); plt.close(fig)

    print(f"\n> графики сохранены в '{args.out_dir}/'")


if __name__ == "__main__":
    main()
