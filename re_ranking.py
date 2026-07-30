"""k-reciprocal encoding re-ranking for person re-identification.

Implementation of

    Zhong, Zheng, Cao, Li — "Re-ranking Person Re-identification with k-reciprocal
    Encoding", CVPR 2017 (arXiv:1701.08398)

following the algorithm of the authors' reference release
(``zhunzhong07/person-reid-triplet-loss-baseline``, ``re_ranking.py``).

The idea: if two images are each other's *k*-nearest neighbours (they are
"k-reciprocal"), they are far more likely to be the same identity than if the
relation only holds in one direction. Each image is encoded as a sparse vector
over its expanded k-reciprocal set, and the images are re-ranked by the Jaccard
distance between those vectors, blended with the original distance.

It is a pure post-processing step on features that have already been extracted:
no training, no extra forward passes.

Memory note
-----------
The algorithm materialises a few ``(n_query + n_gallery)^2`` matrices. For the
19,679-image test gallery plus 2,248 queries that is ~21.9k x 21.9k, i.e. about
0.9 GB per matrix in float16 and ~3 GB in total. On the validation split
(~2.4k gallery) it is negligible. Use ``float32=False`` (the default) to keep the
intermediate matrices in float16.
"""

from __future__ import annotations

import numpy as np

__all__ = ['re_ranking', 'k_reciprocal_neighbours']


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def k_reciprocal_neighbours(initial_rank: np.ndarray, i: int, k: int) -> np.ndarray:
    """Indices that are within the top-k of ``i`` *and* have ``i`` in their own top-k."""
    forward = initial_rank[i, :k + 1]
    backward = initial_rank[forward, :k + 1]
    hit = np.where(backward == i)[0]
    return forward[hit]


def re_ranking(query_features, gallery_features, k1: int = 20, k2: int = 6,
               lambda_value: float = 0.3, float32: bool = False) -> np.ndarray:
    """Return the re-ranked ``(n_query, n_gallery)`` distance matrix (lower = closer).

    Parameters
    ----------
    query_features, gallery_features
        ``(n, d)`` arrays or torch tensors. **They must be L2-normalised** — the
        original distance is computed as ``1 - cosine similarity``.
    k1
        Size of the k-reciprocal neighbourhood. 20 is the value used in the paper
        for Market-1501; lower it (6-10) when the gallery is small, as it is on
        our validation split.
    k2
        Local query expansion size. ``k2 = 1`` disables the expansion.
    lambda_value
        Blend between the Jaccard distance and the original one:
        ``final = (1 - lambda) * jaccard + lambda * original``. ``lambda = 0``
        uses the Jaccard distance alone, ``lambda = 1`` disables re-ranking.
    """
    dtype = np.float32 if float32 else np.float16

    q = _to_numpy(query_features)
    g = _to_numpy(gallery_features)
    n_query = q.shape[0]
    n_all = n_query + g.shape[0]

    feats = np.concatenate([q, g], axis=0)

    # Original distance over the joint set, column-normalised as in the reference
    # implementation (it makes lambda comparable across datasets).
    original_dist = 1.0 - feats @ feats.T
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    # ---- encode each image as a sparse vector over its expanded k-reciprocal set
    V = np.zeros((n_all, n_all), dtype=dtype)
    half_k1 = int(round(k1 / 2)) + 1

    for i in range(n_all):
        k_reciprocal = k_reciprocal_neighbours(initial_rank, i, k1)
        expansion = k_reciprocal

        # a neighbour's own (smaller) k-reciprocal set is merged in when the two
        # sets overlap enough: this recovers positives that the plain
        # neighbourhood misses because of a hard viewpoint change
        for candidate in k_reciprocal:
            cand_forward = initial_rank[candidate, :half_k1]
            cand_backward = initial_rank[cand_forward, :half_k1]
            cand_hit = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_forward[cand_hit]

            overlap = np.intersect1d(cand_reciprocal, k_reciprocal, assume_unique=False)
            if len(overlap) > 2.0 / 3.0 * len(cand_reciprocal):
                expansion = np.append(expansion, cand_reciprocal)

        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        V[i, expansion] = (weight / np.sum(weight)).astype(dtype)

    original_dist = original_dist[:n_query]

    # ---- local query expansion: average each vector with its k2 nearest ones
    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(n_all):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    # ---- Jaccard distance, computed sparsely through an inverted index
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(n_all)]

    jaccard_dist = np.zeros_like(original_dist, dtype=dtype)
    for i in range(n_query):
        intersection = np.zeros(n_all, dtype=dtype)
        non_zero = np.where(V[i] != 0)[0]
        for j in non_zero:
            rows = inv_index[j]
            intersection[rows] += np.minimum(V[i, j], V[rows, j])
        jaccard_dist[i] = 1 - intersection / (2.0 - intersection)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return np.asarray(final_dist[:, n_query:], dtype=np.float32)
