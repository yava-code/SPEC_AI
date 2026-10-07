import os
import time
import random
import numpy as np
import pandas as pd
import psutil
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

process = psutil.Process()


def get_peak_memory_mb(func, *args, **kwargs):
    mem_before = process.memory_info().rss / (1024 * 1024)
    res = func(*args, **kwargs)
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_diff = max(0.0, mem_after - mem_before)
    return res, peak_diff



X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED)
}

results_dir = "repo/lab01/results"
os.makedirs(results_dir, exist_ok=True)

accuracy_records = []
summary_records = []

single_sample = X_test[0:1]

for name, model in models.items():

    model.fit(X_train, y_train)  # warm-up

    train_times = []
    train_mem_diffs = []
    for _ in range(5):
        t0 = time.perf_counter()
        _, mem_diff = get_peak_memory_mb(model.fit, X_train, y_train)
        train_times.append(time.perf_counter() - t0)
        train_mem_diffs.append(mem_diff)

    median_train_time_sec = float(np.median(train_times))
    peak_train_mem_mb = float(np.max(train_mem_diffs))


    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 4)
    accuracy_records.append({"model": name, "test_accuracy": acc})


    model.predict(single_sample)  # warm-up
    inf_times_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(single_sample)
        inf_times_ms.append((time.perf_counter() - t0) * 1000.0)
    median_inf_latency_ms = float(np.median(inf_times_ms))


    model_path = os.path.join(results_dir, f"{name.lower()}.joblib")
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024.0

    summary_records.append({
        "Model": name,
        "Accuracy": acc,
        "Train Time (s)": round(median_train_time_sec, 5),
        "Peak Train RAM (MB)": round(peak_train_mem_mb, 4),
        "Inference Latency (ms)": round(median_inf_latency_ms, 4),
        "Model Size (Bytes)": size_bytes,
        "Model Size (KB)": round(size_kb, 2)
    })


df_acc = pd.DataFrame(accuracy_records)
df_acc.to_csv(os.path.join(results_dir, "baseline_accuracy.csv"), index=False)


df_summary = pd.DataFrame(summary_records)
print("\n" + "=" * 50)
print("BENCHMARK SUMMARY:")
print("=" * 50)
print(df_summary.to_string(index=False))