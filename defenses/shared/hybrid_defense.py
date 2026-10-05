"""Query-aware lexical/vector ranking. No document trust labels are consumed."""

import re
import unicodedata

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower())


def document_tokens(document):
    # Preserve the stored title/text representation, including repeated text.
    return tokenize(str(document.get("title", "")) + " " + str(document.get("text", "")))


def ordered(scores):
    return sorted(scores, key=lambda did: (scores[did], did), reverse=True)


def lexical_scores(index, ids, query):
    scores = index.get_scores(tokenize(query))
    return {did: float(score) for did, score in zip(ids, scores) if score > 0}


def build_index(tokenized_documents):
    return BM25Okapi(tokenized_documents, k1=1.5, b=0.75, epsilon=0.25)


def reciprocal_rank_fusion(dense, lexical, rank_constant=60, window=1000):
    if rank_constant < 1 or window < 1:
        raise ValueError("rank_constant and window must be positive")
    scores = {}
    for channel in (dense, lexical):
        for rank, did in enumerate(ordered(channel)[:window], start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (rank_constant + rank)
    return scores


def lexical_consensus(dense, lexical, window=100):
    if window < 1:
        raise ValueError("window must be positive")
    allowed = set(ordered(lexical)[:window])
    return {did: score for did, score in dense.items() if did in allowed}


def rank_variants(dense, lexical):
    return {
        "dense": dense,
        "bm25_only": lexical,
        "rrf_60": reciprocal_rank_fusion(dense, lexical),
        "consensus_bm25_100": lexical_consensus(dense, lexical),
    }
