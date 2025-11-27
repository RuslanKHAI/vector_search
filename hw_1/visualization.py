import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-GUI backend
import os
import time
import warnings

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

warnings.filterwarnings("ignore")


# Загрузка кэшированных эмбеддингов
def load_embeddings(embeddings_file: str = "hw_1/data/embeddings.npy") -> np.ndarray:
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(
            f"Эмбеддинги не найдены по адресу {embeddings_file}."
            "Пожалуйста, сначала выполните hw_1/search.py для генерации эмбеддингов."
        )
    print(f"Загрузка эмбеддингов из файла: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"Эмбеддинги загружены, форма массива: {embeddings.shape}")
    return embeddings


# Снижение размерности с использованием выбранного метода.
def reduce_dimensionality(
    embeddings: np.ndarray, method: str = "pca", **kwargs
) -> tuple[np.ndarray, float]:
    n_samples = len(embeddings)
    start_time = time.time()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42, **kwargs)
    elif method == "tsne":
        tsne_params = {
            "n_components": 2,
            "random_state": 42,
            "perplexity": min(30, n_samples - 1),
            "max_iter": 1000,
            "n_jobs": -1,
        }
        tsne_params.update(kwargs)
        reducer = TSNE(**tsne_params)
    elif method == "umap":
        umap_params = {
            "n_components": 2,
            "random_state": 42,
            "n_neighbors": min(15, n_samples - 1),
            "min_dist": 0.1,
            "n_jobs": 1,
        }
        umap_params.update(kwargs)
        reducer = UMAP(**umap_params)
    else:
        raise ValueError(f"Неизвестный метод: {method}")  # в случае ошибки
    print(f"Выполняется {method.upper()}... ", end="", flush=True)
    reduced = reducer.fit_transform(embeddings)
    elapsed = time.time() - start_time
    print(f"{elapsed:.2f}s ({n_samples / elapsed:.0f} образцов/сек)")
    return reduced, elapsed


# Визуализация всех трех методов (PCA, t-SNE, UMAP) для заданного размера выборки
def visualize_all_methods_for_size(
    embeddings: np.ndarray, n_samples: int, output_dir: str = "hw_1/results"
) -> dict[str, tuple[np.ndarray, float]]:
    print(f"\n{'=' * 80}")
    print(f"Размер выборки: {n_samples:,}")
    print(f"{'=' * 80}")

    if n_samples < len(embeddings):
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_subset = embeddings[indices]
    else:
        embeddings_subset = embeddings
        n_samples = len(embeddings)
    print(f"Исходная размерность: {embeddings_subset.shape[1]}")

    # Запускаем все методы
    methods = ["pca", "tsne", "umap"]
    results = {}
    for method in methods:
        try:
            reduced, elapsed = reduce_dimensionality(embeddings_subset, method=method)
            results[method] = (reduced, elapsed)
        except Exception as e:
            print(f"  ✗ Error with {method.upper()}: {e}")
            results[method] = (None, None)
    # Создаем визуализацию со всеми тремя методами
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Сравнение методов снижения размерности (n={n_samples:,} образцов)",
        fontsize=16,
        fontweight="bold",
    )
    for idx, method in enumerate(methods):
        ax = axes[idx]
        reduced, elapsed = results[method]
        if reduced is not None:
            # график
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1], alpha=0.6, s=10, c=range(len(reduced)), cmap="viridis"
            )
            # Заголовок
            ax.set_title(
                f"{method.upper()}\n{elapsed:.2f}s ({n_samples / elapsed:.0f} образцов/сек)",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Component 1", fontsize=12)
            ax.set_ylabel("Component 2", fontsize=12)
            ax.grid(True, alpha=0.3)
            # цветовая шкала
            plt.colorbar(scatter, ax=ax, label="Индекс образца")
        else:
            ax.text(
                0.5,
                0.5,
                f"{method.upper()}\nFailed",
                ha="center",
                va="center",
                fontsize=16,
                color="red",
            )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/task_2_2_comparison_{n_samples}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сохранен в файл: {save_path}")
    plt.close(fig)
    return results


# Сравненим PCA, t-SNE и UMAP для разных размеров выборки
def compare_methods_across_sizes(
    embeddings: np.ndarray, sample_sizes: list[int] = None, output_dir: str = "hw_1/results"
):
    if sample_sizes is None:
        sample_sizes = [500, 5000, 25000]
    sample_sizes = [s for s in sample_sizes if s <= len(embeddings)]
    print("\n" + "=" * 80)
    print("Сравнение методов снижения размерности:")
    print("=" * 80)
    print(f"Размеры выборок: {sample_sizes}")
    timing_results = {
        "pca": {"sizes": [], "times": []},
        "tsne": {"sizes": [], "times": []},
        "umap": {"sizes": [], "times": []},
    }
    os.makedirs(output_dir, exist_ok=True)
    # Запуск экспериментов для каждого размера
    for n_samples in sample_sizes:
        results = visualize_all_methods_for_size(embeddings, n_samples, output_dir)
        for method, (reduced, elapsed) in results.items():
            if reduced is not None:
                timing_results[method]["sizes"].append(n_samples)
                timing_results[method]["times"].append(elapsed)
    # Создадим график сравнения времени
    create_timing_comparison_plots(timing_results, output_dir)
    # Создадим сводную таблицу
    create_summary_table(timing_results, output_dir)


# Создадим график сравнения времени выполнения
def create_timing_comparison_plots(results: dict, output_dir: str):
    print("\n" + "=" * 80)
    print("Создаем графики сравнения времени выполнения")
    print("=" * 80)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    for method, data in results.items():
        if data["times"]:
            ax1.plot(
                data["sizes"], data["times"], "o-", label=method.upper(), linewidth=2, markersize=8
            )
    ax1.set_xlabel("Количество образцов", fontsize=12)
    ax1.set_ylabel("Время выполнения (сек)", fontsize=12)
    ax1.set_title(
        "Время выполнения в зависимости от размера выборки", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = axes[1]
    for method, data in results.items():
        if data["times"]:
            throughput = [s / t for s, t in zip(data["sizes"], data["times"])]
            ax2.plot(
                data["sizes"], throughput, "o-", label=method.upper(), linewidth=2, markersize=8
            )

    ax2.set_xlabel("Количество образцов", fontsize=12)
    ax2.set_ylabel("Пропускная способность (образцы/сек)", fontsize=12)
    ax2.set_title("Пропускная способность обработки", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/task_2_2_timing_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Графики сравнения времени сохранены в: {save_path}")
    plt.close(fig)


# Создание и сохранение сводной таблицы
def create_summary_table(results: dict, output_dir: str):
    print("\n" + "=" * 80)
    print("Сводная таблица: Время выполнения")
    print("=" * 80)

    header = f"{'Sample Size':<15} {'PCA (s)':<15} {'t-SNE (s)':<15} {'UMAP (s)':<15}"
    print(header)
    print("-" * 60)

    all_sizes = sorted(
        set(size for method_data in results.values() for size in method_data["sizes"])
    )
    table_lines = ["Сравнение методов снижения размерности", "=" * 80, "", header, "-" * 60]
    for size in all_sizes:
        row = f"{size:<15,}"
        for method in ["pca", "tsne", "umap"]:
            if size in results[method]["sizes"]:
                idx = results[method]["sizes"].index(size)
                time_val = results[method]["times"][idx]
                row += f" {time_val:<15.2f}"
            else:
                row += f" {'N/A':<15}"
        print(row)
        table_lines.append(row)

    table_lines.extend(
        [
            "",
            "",
            "Пропускная способность(образцы/сек):",
            "-" * 60,
            f"{'Sample Size':<15} {'PCA':<15} {'t-SNE':<15} {'UMAP':<15}",
            "-" * 60,
        ]
    )
    print("\n" + "Пропускная способность(образцы/сек):")
    print("-" * 60)
    print(f"{'Размер выборки':<15} {'PCA':<15} {'t-SNE':<15} {'UMAP':<15}")
    print("-" * 60)
    for size in all_sizes:
        row = f"{size:<15,}"
        for method in ["pca", "tsne", "umap"]:
            if size in results[method]["sizes"]:
                idx = results[method]["sizes"].index(size)
                throughput = size / results[method]["times"][idx]
                row += f" {throughput:<15.0f}"
            else:
                row += f" {'N/A':<15}"
        print(row)
        table_lines.append(row)

    with open(f"{output_dir}/task_2_2_summary_table.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
        f.write("\n\n")
        f.write("Примечание: время указано в секундах\n")

    print(f"\nСводная таблица сохранена в файл: {output_dir}/task_2_2_summary_table.txt")


# Запуск всех экспериментов по визуализации
def run_visualization_experiments():
    print("=" * 80)
    print("Визуализация снижения размерности")
    print("=" * 80)
    # Загрузка эмбеддингов
    embeddings = load_embeddings()
    sample_sizes = [500, 5000, 25000]
    sample_sizes = [s for s in sample_sizes if s <= len(embeddings)]
    if not sample_sizes:
        print("Ошибка: Недостаточно эмбеддингов!")
        return
    print(f"\nТестирование на размерах выборок: {sample_sizes}")
    print(f"Доступно эмбеддингов: {len(embeddings):,}")
    compare_methods_across_sizes(embeddings, sample_sizes=sample_sizes, output_dir="hw_1/results")

    print("\nСгенерированные файлы:")
    print("  • task_2_2_omparison_500.png - Все методы с 500 образцами")
    print("  • task_2_2_comparison_5000.png - Все методы с 5,000 образцами")
    print("  • task_2_2_comparison_25000.png - Все методы с 25,000 образцами")
    print("  • task_2_2_timing_comparison.png - Сравнение скорости работы методов")
    print("  • task_2_2_summary_table.txt - результаты времени")
    print("\nВсе результаты сохранены в: hw_1/results/")


if __name__ == "__main__":
    run_visualization_experiments()
