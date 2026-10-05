"""Bounded dense-first lexical reranking and weighted rank fusion."""

from hybrid_defense import ordered, tokenize


def dense_shortlist(dense, limit=100):
    if limit < 1:
        raise ValueError("limit must be positive")
    return ordered(dense)[:limit]


def subset_bm25(index, id_to_position, query, candidates):
    scores = index.get_batch_scores(tokenize(query), [id_to_position[d] for d in candidates])
    return dict(zip(candidates, scores))


def reverse_consensus(dense, lexical, limit=100):
    candidates = dense_shortlist(dense, limit)
    # Dense breaks lexical ties, including the all-zero lexical fallback.
    ranked = sorted(candidates, key=lambda d: (lexical[d], dense[d], d), reverse=True)
    return {did: float(len(ranked) - rank) for rank, did in enumerate(ranked)}


def budgeted_fusion(dense, lexical, limit=100, lexical_weight=0.75, constant=60):
    if not 0 <= lexical_weight <= 1 or constant < 1:
        raise ValueError("Invalid fusion parameters")
    candidates = dense_shortlist(dense, limit)
    positive = sorted((d for d in candidates if lexical[d] > 0),
                      key=lambda d: (lexical[d], dense[d], d), reverse=True)
    if not positive:
        return {d: dense[d] for d in candidates}
    lexical_ranks = {d: rank for rank, d in enumerate(positive, 1)}
    return {
        d: (1 - lexical_weight) / (constant + rank)
        + (lexical_weight / (constant + lexical_ranks[d]) if d in lexical_ranks else 0)
        for rank, d in enumerate(candidates, 1)
    }
