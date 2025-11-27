import numpy as np

# Pure numpy
# ---------------------


# Compute dot product using pure numpy.
def dot_product_numpy(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        return queries @ matrix.T
    return queries @ matrix.T


# Compute cosine similarity using pure numpy
def cosine_similarity_numpy(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    result = queries_norm @ matrix_norm.T
    return result.squeeze() if squeeze else result


# Compute Euclidean distance using pure numpy
def euclidean_distance_numpy(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    queries_sq = np.sum(queries**2, axis=1, keepdims=True)
    matrix_sq = np.sum(matrix**2, axis=1)
    cross_term = queries @ matrix.T
    result = np.sqrt(np.maximum(queries_sq + matrix_sq - 2 * cross_term, 0))
    return result.squeeze() if squeeze else result


# Compute Manhattan distance using pure numpy
def manhattan_distance_numpy(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    result = np.sum(np.abs(queries[:, np.newaxis, :] - matrix[np.newaxis, :, :]), axis=2)
    return result.squeeze() if squeeze else result


# pure Python
# ----------------------


# Compute dot product using pure Python
def dot_product_python(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = [queries.tolist()]
        squeeze = True
    else:
        queries = queries.tolist()
        squeeze = False
    matrix = matrix.tolist()
    result = []
    for query in queries:
        row = []
        for vec in matrix:
            dot = sum(q * v for q, v in zip(query, vec))
            row.append(dot)
        result.append(row)
    result = np.array(result)
    return result.squeeze() if squeeze else result


# Compute cosine similarity using pure Python
def cosine_similarity_python(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = [queries.tolist()]
        squeeze = True
    else:
        queries = queries.tolist()
        squeeze = False
    matrix = matrix.tolist()
    result = []
    for query in queries:
        query_norm = sum(q**2 for q in query) ** 0.5
        query_normalized = [q / query_norm for q in query]
        row = []
        for vec in matrix:
            vec_norm = sum(v**2 for v in vec) ** 0.5
            vec_normalized = [v / vec_norm for v in vec]
            similarity = sum(q * v for q, v in zip(query_normalized, vec_normalized))
            row.append(similarity)
        result.append(row)
    result = np.array(result)
    return result.squeeze() if squeeze else result


# Compute euclidean distance using pure Pytho
def euclidean_distance_python(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = [queries.tolist()]
        squeeze = True
    else:
        queries = queries.tolist()
        squeeze = False
    matrix = matrix.tolist()
    result = []
    for query in queries:
        row = []
        for vec in matrix:
            distance = sum((q - v) ** 2 for q, v in zip(query, vec)) ** 0.5
            row.append(distance)
        result.append(row)
    result = np.array(result)
    return result.squeeze() if squeeze else result


# Compute Manhattan distance using pure Python
def manhattan_distance_python(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if queries.ndim == 1:
        queries = [queries.tolist()]
        squeeze = True
    else:
        queries = queries.tolist()
        squeeze = False
    matrix = matrix.tolist()
    result = []
    for query in queries:
        row = []
        for vec in matrix:
            distance = sum(abs(q - v) for q, v in zip(query, vec))
            row.append(distance)
        result.append(row)
    result = np.array(result)
    return result.squeeze() if squeeze else result
