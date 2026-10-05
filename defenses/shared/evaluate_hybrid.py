"""Evaluate BM25/RRF defenses on saved Kaggle scores, including clean controls."""

import argparse
import hashlib
import importlib.metadata
import time
from pathlib import Path

from evaluate_defenses import aggregate, read_corpus, read_json, read_qrels, read_run, write_json
from hybrid_defense import build_index, document_tokens, lexical_scores, rank_variants


def compact(metrics):
    return {key: value for key, value in metrics.items() if key != "per_query"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "results.json").exists():
        parser.error("Choose a fresh output directory; results.json already exists")
    start = time.perf_counter()
    base = args.root / "usenix_1226_AE/results/scifact/gte_modernbert_base"
    attack = base / "poisoned/10_5000_30_0.2_0.55_prefix_42"
    dataset = args.root / "usenix_1226_AE_work_gte/datasets/scifact"
    corpus = read_corpus(dataset / "corpus.jsonl")
    poison = read_json(attack / "poisoned_added_corpus.json")
    queries = read_json(attack / "tested_queries.json")
    target_labels = read_json(attack / "poisoned_qrels.json")
    qrels = read_qrels(dataset / "qrels/test.tsv")
    dense_clean = read_run(base / "clean_results.run.trec")
    dense_poison = read_run(attack / "poisoned_results.run.trec")
    if not queries or set(corpus) & set(poison):
        raise ValueError("Empty query set or colliding document IDs")
    targets = {}
    for qid in queries:
        if len(target_labels[qid]) != 1:
            raise ValueError("Expected exactly one attack document per query")
        target = next(iter(target_labels[qid]))
        if target not in poison or target not in dense_poison[qid] or not dense_clean[qid]:
            raise ValueError(f"Missing attack document or scores: {qid}")
        if qid not in qrels:
            raise ValueError(f"Missing relevance labels: {qid}")
        targets[qid] = target
    if set(targets.values()) != set(poison):
        raise ValueError("Target document coverage mismatch")

    ids = list(corpus)
    tokens = [document_tokens(corpus[did]) for did in ids]
    clean_index = build_index(tokens)
    scenarios = {name: {} for name in ("clean", "attacked")}
    for i, (qid, query) in enumerate(queries.items(), start=1):
        target = targets[qid]
        clean_scores = {did: score for did, score in dense_clean[qid].items() if did != qid}
        # Each scenario inserts only its target, matching the original evaluation.
        attack_index = build_index(tokens + [document_tokens(poison[target])])
        attack_dense = {**clean_scores, target: dense_poison[qid][target]}
        clean_lexical = lexical_scores(clean_index, ids, query)
        attack_lexical = lexical_scores(attack_index, ids + [target], query)
        clean_lexical.pop(qid, None)
        attack_lexical.pop(qid, None)
        for scenario, dense, lexical in (("clean", clean_scores, clean_lexical),
                                         ("attacked", attack_dense, attack_lexical)):
            for name, scores in rank_variants(dense, lexical).items():
                scenarios[scenario].setdefault(name, {})[qid] = scores
        if i % 10 == 0:
            print(f"Evaluated {i}/{len(queries)} queries in {time.perf_counter()-start:.1f}s", flush=True)

    report = {
        "query_count": len(queries), "corpus_count": len(corpus),
        "attack_document_count": len(poison), "k": 5,
        "settings": {"bm25_k1": 1.5, "bm25_b": 0.75, "bm25_epsilon": 0.25,
                     "rrf_rank_constant": 60, "rrf_window": 1000,
                     "consensus_bm25_window": 100,
                     "tokenization": "NFKC, lowercase, Unicode word tokens; no stemming or stopword removal",
                     "selection": "Fixed settings before first run; no parameter sweep or tuning on these queries",
                     "lexical_scope": "Full 5183-document clean corpus plus one target in each attacked scenario",
                     "dense_scope": "Saved top-1000 clean candidates plus the target score"},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "rank-bm25")},
        "scenarios": {},
    }
    for scenario, methods in scenarios.items():
        report["scenarios"][scenario] = {}
        for name, scores in methods.items():
            measured = aggregate(scores, queries, targets, qrels, 5)
            measured["queries_with_fewer_than_k_results"] = sum(len(row) < 5 for row in scores.values())
            report["scenarios"][scenario][name] = measured
    exported = read_json(attack / "poisoned_metrics_poisoned.json")
    mismatch = [qid for qid, row in report["scenarios"]["attacked"]["dense"]["per_query"].items()
                if abs(float(row["target_hit_at_k"]) - exported["recall"][qid]["Recall@5"]) > 1e-5]
    if mismatch:
        raise ValueError(f"Baseline differs from original metrics: {mismatch}")
    report["baseline_attack_metric_mismatches"] = mismatch
    source_files = [dataset / "corpus.jsonl", dataset / "qrels/test.tsv",
                    base / "clean_results.run.trec", attack / "poisoned_results.run.trec",
                    *sorted(attack.glob("*.json")), Path(__file__),
                    Path(__file__).with_name("hybrid_defense.py"),
                    Path(__file__).with_name("evaluate_defenses.py")]
    report["input_sha256_for_reproducibility_only"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(args.output / "results.json", report)
    summary = {key: value for key, value in report.items()
               if key not in {"scenarios", "input_sha256_for_reproducibility_only"}}
    summary["scenarios"] = {scenario: {name: compact(metrics) for name, metrics in methods.items()}
                            for scenario, methods in report["scenarios"].items()}
    write_json(args.output / "summary.json", summary)
    for scenario, methods in summary["scenarios"].items():
        for name, metrics in methods.items():
            print(scenario, name, metrics)


if __name__ == "__main__":
    main()
