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
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.metrics.pairwise import euclidean_distances as sklearn_euclidean
from sklearn.metrics.pairwise import manhattan_distances as sklearn_manhattan


# Пртестируем cosine similarity с sklearn
def test_cosine_similarity():
    print("\nТест Cosine Similarity:")
    np.random.seed(42)
    query = np.random.rand(5)
    queries = np.random.rand(10, 5)
    data = np.random.rand(100, 5)

    # numpy - single query
    result = cosine_similarity_numpy(query.copy(), data.copy())
    expected = sklearn_cosine(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected), "Single query - Numpy отличается от sklearn"
    print("Реализация Numpy (single query) совпадаютsklearn")

    # numpy - multiple queries
    result = cosine_similarity_numpy(queries.copy(), data.copy())
    expected = sklearn_cosine(queries, data)
    assert np.allclose(result, expected), "Multiple queries - Numpy отличается от sklearn "
    print("Реализация Numpy (multiple queries) совпадаютsklearn")

    # python - single query
    result = cosine_similarity_python(query.copy(), data.copy())
    expected = sklearn_cosine(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected, rtol=1e-5), "Single query-Python отличается от sklearn"
    print("Реализация Pure Python (single query) совпадают sklearn")

    # python - multiple queries
    result = cosine_similarity_python(queries.copy(), data.copy())
    expected = sklearn_cosine(queries, data)
    assert np.allclose(
        result, expected, rtol=1e-5
    ), "Multiple queries -Python отличается от sklearn"
    print("Реализация Pure Python (multiple queries) совпадают sklearn")

    # numpy vs python
    np_result = cosine_similarity_numpy(query.copy(), data.copy())
    py_result = cosine_similarity_python(query.copy(), data.copy())
    assert np.allclose(np_result, py_result, rtol=1e-5), "Numpy и Python различаются"
    print("Реализация Numpy and Python совпадают")


# Пртестируем Euclidean distance c sklearn
def test_euclidean_distance():
    print("\nТест Euclidean Distance:")
    np.random.seed(42)
    query = np.random.rand(5)
    queries = np.random.rand(10, 5)
    data = np.random.rand(100, 5)

    # numpy - single query
    result = euclidean_distance_numpy(query.copy(), data.copy())
    expected = sklearn_euclidean(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected), "Single query - Numpy отличается от sklearn"
    print("Реализация Numpy (single query)совпадаютsklearn")

    # numpy - multiple queries
    result = euclidean_distance_numpy(queries.copy(), data.copy())
    expected = sklearn_euclidean(queries, data)
    assert np.allclose(result, expected), "Multiple queries - Numpy отличается от sklearn"
    print("Реализация Numpy (multiple queries) совпадаютsklearn")

    # python - single query
    result = euclidean_distance_python(query.copy(), data.copy())
    expected = sklearn_euclidean(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected, rtol=1e-5), "Single query-Python отличается от sklearn"
    print("Реализация Pure Python (single query) совпадают sklearn")

    # python - multiple queries
    result = euclidean_distance_python(queries.copy(), data.copy())
    expected = sklearn_euclidean(queries, data)
    assert np.allclose(
        result, expected, rtol=1e-5
    ), "Multiple queries -Python отличается от sklearn"
    print("Реализация Pure Python (multiple queries) совпадают sklearn")

    # numpy vs python
    np_result = euclidean_distance_numpy(query.copy(), data.copy())
    py_result = euclidean_distance_python(query.copy(), data.copy())
    assert np.allclose(np_result, py_result, rtol=1e-5), "Numpy и Python различаются"
    print("Реализация Numpy and Python совпадают")


# Пртестируем Manhattan distance c sklearn
def test_manhattan_distance():
    print("\nТест Manhattan Distance:")
    np.random.seed(42)
    query = np.random.rand(5)
    queries = np.random.rand(10, 5)
    data = np.random.rand(100, 5)

    # numpy - single query
    result = manhattan_distance_numpy(query.copy(), data.copy())
    expected = sklearn_manhattan(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected), "Single query - Numpy отличается от sklearn"
    print("Реализация Numpy (single query)совпадаютsklearn")

    # numpy - multiple queries
    result = manhattan_distance_numpy(queries.copy(), data.copy())
    expected = sklearn_manhattan(queries, data)
    assert np.allclose(result, expected), "Multiple queries - Numpy отличается от sklearn"
    print("Реализация Numpy (multiple queries) совпадаютsklearn")

    # python - single query
    result = manhattan_distance_python(query.copy(), data.copy())
    expected = sklearn_manhattan(query.reshape(1, -1), data).squeeze()
    assert np.allclose(result, expected, rtol=1e-5), "Single query-Python отличается от sklearn"
    print("Реализация Pure Python (multiple queries) совпадают sklearn")

    # python - multiple queries
    result = manhattan_distance_python(queries.copy(), data.copy())
    expected = sklearn_manhattan(queries, data)
    assert np.allclose(
        result, expected, rtol=1e-5
    ), "Multiple queries -Python отличается от sklearn"
    print("Реализация Pure Python (multiple queries) совпадают sklearn")

    # Compare numpy vs python
    np_result = manhattan_distance_numpy(query.copy(), data.copy())
    py_result = manhattan_distance_python(query.copy(), data.copy())
    assert np.allclose(np_result, py_result, rtol=1e-5), "Numpy и Python различаются"
    print("Реализация Numpy and Python совпадают")


# Пртестируем dot product implementations
def test_dot_product():
    print("\nTesting Dot Product:")
    np.random.seed(42)
    query = np.random.rand(5)
    queries = np.random.rand(10, 5)
    data = np.random.rand(100, 5)

    # numpy vs python - single query
    np_result = dot_product_numpy(query.copy(), data.copy())
    py_result = dot_product_python(query.copy(), data.copy())
    assert np.allclose(
        np_result, py_result, rtol=1e-5
    ), "Single query: Numpy and Python различаются"
    print("Реализация Numpy and Python совпадают(single query)")

    # numpy vs python - multiple queries
    np_result = dot_product_numpy(queries.copy(), data.copy())
    py_result = dot_product_python(queries.copy(), data.copy())
    assert np.allclose(np_result, py_result, rtol=1e-5), "Multiple: Numpy and Python differ"
    print("Реализация Numpy and Python совпадают (multiple queries)")


if __name__ == "__main__":
    print("=" * 60)
    print("Distance Metrics Tests:")
    print("=" * 60)

    test_dot_product()
    test_cosine_similarity()
    test_euclidean_distance()
    test_manhattan_distance()

    print("\n" + "=" * 60)
    print("Тесты пройдены!")
    print("=" * 60)
