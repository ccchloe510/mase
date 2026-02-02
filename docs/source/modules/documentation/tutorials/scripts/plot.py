import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_best_curve(filename):
    try:
        df = pd.read_csv(filename)
        values = df["value"].values

        return np.maximum.accumulate(values)
    except FileNotFoundError:
        print(f"Can't find the file: {filename}")
        return []

y1 = get_best_curve("results_baseline.csv")
y2 = get_best_curve("results_compress_no_retrain.csv")
y3 = get_best_curve("results_compress_retrain.csv")

min_len = min(len(y1), len(y2), len(y3))
x = range(1, min_len + 1)

plt.figure(figsize=(10, 6))

plt.plot(x, y1[:min_len], label="Task 1: Baseline (No Compression)", marker="o", linestyle="-")
plt.plot(x, y2[:min_len], label="Task 2: Compress Only (No Retrain)", marker="x", linestyle="--")
plt.plot(x, y3[:min_len], label="Task 2: Compress + Retrain", marker="^", linestyle="-.")

plt.xlabel("Number of Trials")
plt.ylabel("Max Accuracy Achieved")
plt.title("NAS Performance: Baseline vs. Compression-Aware")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig("nas_comparison_curve.png")
plt.show()
print("saved nas_comparison_curve.png")