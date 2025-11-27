import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from distances import (
    cosine_similarity_numpy,
    cosine_similarity_python,
    dot_product_numpy,
    dot_product_python,
    euclidean_distance_numpy,
    euclidean_distance_python,
    manhattan_distance_numpy,
    manhattan_distance_python,
)


def benchmark_function(
    func: Callable, query: np.ndarray, data: np.ndarray, n_runs: int = 5
) -> float:
    times = []
    _ = func(query.copy(), data.copy())

    for _ in range(n_runs):
        start = time.time()
        _ = func(query.copy(), data.copy())
        elapsed = time.time() - start
        times.append(elapsed)
    return np.mean(times)


def benchmark_varying_vectors(
    func_numpy: Callable,
    func_python: Callable,
    dim: int = 100,
    n_vectors_list: list[int] = None,
    n_runs: int = 3,
) -> tuple[list[float], list[float], list[float]]:
    # тест с различным кол-вом векторов.
    if n_vectors_list is None:
        n_vectors_list = [100, 500, 1000, 5000, 10000]

    numpy_times = []
    python_times = []

    print(f"\nСравнительный анализ с размерностью={dim}, изменяющимся кол-вом векторов:")
    print(f"{'N Vectors':<12} {'numpy (s)':<15} {'python (s)':<15} {'Speedup':<10}")
    print("-" * 90)
    for n_vectors in n_vectors_list:
        np.random.seed(42)
        query = np.random.rand(dim)
        data = np.random.rand(n_vectors, dim)
        # Benchmark numpy
        time_numpy = benchmark_function(func_numpy, query, data, n_runs)
        numpy_times.append(time_numpy)
        # Benchmark python
        time_python = benchmark_function(func_python, query, data, n_runs)
        python_times.append(time_python)
        speedup = time_python / time_numpy
        print(f"{n_vectors:<12} {time_numpy:<15.6f} {time_python:<15.6f} {speedup:<10.2f}x")
    return n_vectors_list, numpy_times, python_times


def benchmark_varying_dimensions(
    func_numpy: Callable,
    func_python: Callable,
    n_vectors: int = 1000,
    dim_list: list[int] = None,
    n_runs: int = 3,
) -> tuple[list[int], list[float], list[float]]:
    # Benchmark with varying dimensionality.
    if dim_list is None:
        dim_list = [10, 50, 100, 300, 500]
    numpy_times = []
    python_times = []

    print(f"\nСравнительный анализ с использованием n_vectors={n_vectors}, различной размерности:")
    print(f"{'Dimension':<12} {'numpy (s)':<15} {'python (s)':<15} {'Speedup':<10}")
    print("-" * 60)

    for dim in dim_list:
        np.random.seed(42)
        query = np.random.rand(dim)
        data = np.random.rand(n_vectors, dim)
        # benchmark numpy
        time_numpy = benchmark_function(func_numpy, query, data, n_runs)
        numpy_times.append(time_numpy)
        # benchmark python
        time_python = benchmark_function(func_python, query, data, n_runs)
        python_times.append(time_python)
        speedup = time_python / time_numpy
        print(f"{dim:<12} {time_numpy:<15.6f} {time_python:<15.6f} {speedup:<10.2f}x")
    return dim_list, numpy_times, python_times


def plot_benchmark_results(
    x_values: list,
    numpy_times: list[float],
    python_times: list[float],
    xlabel: str,
    title: str,
    filename: str,
):
    # Plot benchmark results
    plt.figure(figsize=(12, 5))
    # Plot 1: times (s))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, numpy_times, "o-", label="numpy", linewidth=2, markersize=8)
    plt.plot(x_values, python_times, "s-", label="Pure python", linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title(f"{title}\nAbsolute Time", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    # Plot 2: Speedup (не обязательно, но реализуюю)
    plt.subplot(1, 2, 2)
    speedup = [p / n for p, n in zip(python_times, numpy_times)]
    plt.plot(x_values, speedup, "o-", color="green", linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Speedup (python time / numpy time)", fontsize=12)
    plt.title(f"{title}\nnumpy Speedup", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="No speedup")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {filename}")
    plt.close()


# benchmark experiments
def run_all_benchmarks():
    print("=" * 70)
    print("benchmark: pure python vs numpy distance metrics")
    print("=" * 70)
    # Создадим папку results для сохранения резульатов работы кода.
    import os

    os.makedirs("hw_1/results", exist_ok=True)
    # config
    metrics = [
        ("Euclidean", euclidean_distance_numpy, euclidean_distance_python),
        ("Cosine", cosine_similarity_numpy, cosine_similarity_python),
        ("Manhattan", manhattan_distance_numpy, manhattan_distance_python),
        ("Dot Product", dot_product_numpy, dot_product_python),
    ]
    # таблица совсеми данными
    results_vectors = {}
    results_dimensions = {}
    n_vectors_list = [100, 500, 1000, 5000, 10000]
    dim_list = [10, 50, 100, 300, 500]
    dim_for_vectors = 100
    n_for_dimensions = 1000
    for metric_name, func_numpy, func_python in metrics:
        print("\n" + "=" * 70)
        print(f"METRIC: {metric_name}")
        print("=" * 70)
        # Experiment 1: различное кол-во векторов
        n_vec, numpy_times_v, python_times_v = benchmark_varying_vectors(
            func_numpy, func_python, dim=dim_for_vectors, n_vectors_list=n_vectors_list, n_runs=3
        )
        results_vectors[metric_name] = (numpy_times_v, python_times_v)
        plot_benchmark_results(
            n_vec,
            numpy_times_v,
            python_times_v,
            xlabel="Number of vectors",
            title=f"{metric_name}: Varying vector count",
            filename=f"hw_1/results/{metric_name.lower().replace(' ', '_')}_vectors.png",
        )
        # Experiment 2: различные размерности
        dims, numpy_times_d, python_times_d = benchmark_varying_dimensions(
            func_numpy, func_python, n_vectors=n_for_dimensions, dim_list=dim_list, n_runs=3
        )
        results_dimensions[metric_name] = (numpy_times_d, python_times_d)

        plot_benchmark_results(
            dims,
            numpy_times_d,
            python_times_d,
            xlabel="Dimensionality",
            title=f"{metric_name}: Varying dimension",
            filename=f"hw_1/results/{metric_name.lower().replace(' ', '_')}_dimensions.png",
        )
    # summary tables
    print_summary_table_vectors(results_vectors, n_vectors_list, dim_for_vectors)
    print_summary_table_dimensions(results_dimensions, dim_list, n_for_dimensions)

    # Save to file
    save_summary_tables_to_file(
        results_vectors,
        results_dimensions,
        n_vectors_list,
        dim_list,
        dim_for_vectors,
        n_for_dimensions,
    )
    print("\nРезультаты сохранены в: hw_1/results/")


# summary table for varying vectors experiment
def print_summary_table_vectors(results_dict: dict, n_vectors_list: list[int], dim: int):
    print("\n" + "=" * 120)
    print(f"summary table: varying number of vectors (dimension = {dim})")
    print("=" * 120)

    header = f"{'n':<10} {'d':<10}"
    for metric_name in results_dict.keys():
        header += f" {metric_name + ' (numpy)':<20} {metric_name + ' (python)':<20}"
    print(header)
    print("-" * 120)

    for i, n in enumerate(n_vectors_list):
        row = f"{n:<10} {dim:<10}"
        for metric_name, (numpy_times, python_times) in results_dict.items():
            row += f" {numpy_times[i]:<20.6f} {python_times[i]:<20.6f}"
        print(row)
    print("=" * 120)


# summary table for varying dimensions experiment
def print_summary_table_dimensions(results_dict: dict, dim_list: list[int], n_vectors: int):
    print("\n" + "=" * 120)
    print(f"summary table: varying dimensionality (n_vectors = {n_vectors})")
    print("=" * 120)

    header = f"{'n':<10} {'d':<10}"
    for metric_name in results_dict.keys():
        header += f" {metric_name + ' (numpy)':<20} {metric_name + ' (python)':<20}"
    print(header)
    print("-" * 120)

    for i, d in enumerate(dim_list):
        row = f"{n_vectors:<10} {d:<10}"
        for metric_name, (numpy_times, python_times) in results_dict.items():
            row += f" {numpy_times[i]:<20.6f} {python_times[i]:<20.6f}"
        print(row)

    print("=" * 120)


def save_summary_tables_to_file(
    results_vectors: dict,
    results_dimensions: dict,
    n_vectors_list: list[int],
    dim_list: list[int],
    dim_for_vectors: int,
    n_for_dimensions: int,
    filename: str = "hw_1/results/benchmark_summary.txt",
):
    with open(filename, "w", encoding="utf-8") as f:
        # таблица 1 - различные векторы
        f.write("=" * 120 + "\n")
        f.write(f"summary tables 1: varying number of vectors (dimension = {dim_for_vectors})\n")
        f.write("=" * 120 + "\n\n")

        header = f"{'n':<10} {'d':<10}"
        for metric_name in results_vectors.keys():
            header += f" {metric_name + ' (numpy)':<20} {metric_name + ' (python)':<20}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for i, n in enumerate(n_vectors_list):
            row = f"{n:<10} {dim_for_vectors:<10}"
            for metric_name, (numpy_times, python_times) in results_vectors.items():
                row += f" {numpy_times[i]:<20.6f} {python_times[i]:<20.6f}"
            f.write(row + "\n")
        f.write("=" * 120 + "\n\n\n")

        # Таблица2 - Различне размеры
        f.write("=" * 120 + "\n")
        f.write(f"summary tables 2: varying dimensionality (n_vectors = {n_for_dimensions})\n")
        f.write("=" * 120 + "\n\n")

        header = f"{'n':<10} {'d':<10}"
        for metric_name in results_dimensions.keys():
            header += f" {metric_name + ' (numpy)':<20} {metric_name + ' (python)':<20}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        for i, d in enumerate(dim_list):
            row = f"{n_for_dimensions:<10} {d:<10}"
            for metric_name, (numpy_times, python_times) in results_dimensions.items():
                row += f" {numpy_times[i]:<20.6f} {python_times[i]:<20.6f}"
            f.write(row + "\n")
        f.write("=" * 120 + "\n")
    print(f"\nСводные таблицы сохранены: {filename}")


if __name__ == "__main__":
    run_all_benchmarks()
