import os

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# Текстовый поиск с использованием embeddings
class TextSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.texts: list[str] | None = None
        print("Model loaded successfully")

    # Загрузка датасета
    def load_dataset_and_embed(
        self,
        dataset_name: str = "ag_news",
        text_column: str = "text",
        max_samples: int = 50000,
        cache_file: str = "hw_1/data/embeddings.npy",
    ):
        # Создадим папку data для хранения данных
        os.makedirs("hw_1/data", exist_ok=True)
        # загрузка embeddings
        texts_cache = cache_file.replace(".npy", "_texts.npy")
        if os.path.exists(cache_file) and os.path.exists(texts_cache):
            print(f"Загрузка кэшированных эмбеддингов из файла {cache_file}...")
            self.embeddings = np.load(cache_file)
            self.texts = np.load(texts_cache, allow_pickle=True).tolist()
            print(f"Загружено {len(self.texts)} кэшированных embeddings")
            return
        # загрузка dataset
        print(f"Загрузка датасета: {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train")
        # Ограничем выборку
        if len(dataset) > max_samples:
            indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=int)
            dataset = dataset.select(indices)
        print(f"Загружено {len(dataset)} выборки")
        # Извлечение текстов
        self.texts = [item[text_column] for item in dataset]
        # Создадим embeddings
        print("Создание embeddings:")
        self.embeddings = self.model.encode(
            self.texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True
        )
        print(f"Созданные эмбеддинги имеют форму: {self.embeddings.shape}")
        np.save(cache_file, self.embeddings)
        np.save(texts_cache, np.array(self.texts, dtype=object))
        print(f"Эмбеддинги закэшированы в {cache_file}")

    # Поиск ближайших соседей
    # Distance metric ('cosine', 'euclidean', 'manhattan', 'dot')
    def search(
        self,
        query: str,
        k: int = 5,  # Количество результатов для возврата
        metric: str = "cosine",
    ) -> list[tuple[int, float, str]]:
        if self.embeddings is None or self.texts is None:
            raise ValueError("Сначала необходимо загрузить датасет")
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        # Вычисление расстояний
        if metric == "cosine":
            from distances import cosine_similarity_numpy

            scores = cosine_similarity_numpy(query_embedding, self.embeddings)
            # инвертируем для сортировки/Для косинусного сходства больше - лучше
            distances = -scores
        elif metric == "euclidean":
            from distances import euclidean_distance_numpy

            distances = euclidean_distance_numpy(query_embedding, self.embeddings)
        elif metric == "manhattan":
            from distances import manhattan_distance_numpy

            distances = manhattan_distance_numpy(query_embedding, self.embeddings)
        elif metric == "dot":
            from distances import dot_product_numpy

            scores = dot_product_numpy(query_embedding, self.embeddings)
            distances = -scores
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")
        # Получаем top-k
        top_k_indices = np.argsort(distances)[:k]
        # results
        results = [(int(idx), float(distances[idx]), self.texts[idx]) for idx in top_k_indices]
        return results

    # Результат поиска
    def print_search_results(self, query: str, results: list[tuple[int, float, str]], metric: str):
        print("\n" + "=" * 80)
        print(f"Query: '{query}'")
        print(f"Metric: {metric}")
        print("=" * 80)
        # Для косинуса и скалярного произведения покажем положительные значения
        for rank, (idx, distance, text) in enumerate(results, 1):
            score_str = f"{-distance:.4f}" if metric in ["cosine", "dot"] else f"{distance:.4f}"
            # Обрезаем длинные тексты
            display_text = text if len(text) <= 100 else text[:97] + "..."
            print(f"\n[{rank}] Score: {score_str}")
            print(f"    {display_text}")
        print("=" * 80)


# Запуск экспериментов поиска по примерам запросов
def run_search_experiments():
    print("=" * 80)
    print("Эксперементы по поиску по текстовой схожести")
    print("=" * 80)
    # Поисковый движжок
    engine = TextSearchEngine(model_name="all-MiniLM-L6-v2")
    # Загружаем датасет и создадим эмбединги.
    # используем  датасет:AG News (120k news articles in 4 categories)
    engine.load_dataset_and_embed(dataset_name="ag_news", text_column="text", max_samples=50000)
    # ПРимеры запросов
    queries = [
        "Technology and artificial intelligence",
        "Global warming and climate change",
        "Politics and elections",
        "Sports and football championship",
        "Economy and stock market crash",
        "Health and medical research",
        "Entertainment and Hollywood movies",
        "Science and space exploration",
        "Business mergers and acquisitions",
        "International conflicts and wars",
    ]

    # Тестируем разыне метрики
    metrics = ["cosine", "euclidean", "manhattan", "dot"]

    print("\n" + "=" * 80)
    print("Сравниваем разыне метрики растояния")
    print("=" * 80)
    test_query = queries[0]
    print(f"\nТестовый запрос: '{test_query}'")

    for metric in metrics:
        results = engine.search(test_query, k=5, metric=metric)
        engine.print_search_results(test_query, results, metric)

    print("\n" + "=" * 80)
    print("Результаты поиска по запросам (cosine similarity)")
    print("=" * 80)

    for query in queries[1:]:
        results = engine.search(query, k=5, metric="cosine")
        engine.print_search_results(query, results, "cosine")

    # сохранение результатов в файл
    print("\nСохранение подробных результатов в файл")
    os.makedirs("hw_1/results", exist_ok=True)
    with open("hw_1/results/search_results.txt", "w", encoding="utf-8") as f:
        f.write("Результат поиска по текстовой схожести\n")
        f.write("Датасет: AG News (50,000 news articles)\n")
        f.write("Model: all-MiniLM-L6-v2\n")
        f.write("=" * 80 + "\n\n")
        for query in queries:
            results = engine.search(query, k=5, metric="cosine")
            f.write(f"Query: '{query}'\n")
            f.write("-" * 80 + "\n")
            for rank, (idx, distance, text) in enumerate(results, 1):
                f.write(f"[{rank}] Score: {-distance:.4f}\n")
                f.write(f"    {text}\n\n")
            f.write("\n")
    print("Результаты сохранены в файл: hw_1/results/search_results.txt")
    print("\n" + "=" * 80)
    print("Стаистика датасета")
    print("=" * 80)
    print(f"Всего документов: {len(engine.texts)}")
    print(f"Размерность эмбеддингов: {engine.embeddings.shape[1]}")
    print(f"Средняя длина текста: {np.mean([len(t) for t in engine.texts]):.1f} символов")


if __name__ == "__main__":
    run_search_experiments()
