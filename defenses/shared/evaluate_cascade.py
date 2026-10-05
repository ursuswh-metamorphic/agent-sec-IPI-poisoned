"""Dense-first defense replay with clean controls and incremental CPU timings."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cascade_defense import budgeted_fusion, dense_shortlist, reverse_consensus, subset_bm25
from evaluate_defenses import aggregate, read_corpus, read_json, read_qrels, read_run, write_json
from hybrid_defense import build_index, document_tokens, lexical_consensus, lexical_scores, reciprocal_rank_fusion


def stats(values):
    return {"samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "results.json").exists():
        parser.error("Choose a new output directory")
    started = time.perf_counter()
    base = args.root / "usenix_1226_AE/results/scifact/gte_modernbert_base"
    attack = base / "poisoned/10_5000_30_0.2_0.55_prefix_42"
    dataset = args.root / "usenix_1226_AE_work_gte/datasets/scifact"
    corpus = read_corpus(dataset / "corpus.jsonl")
    poison = read_json(attack / "poisoned_added_corpus.json")
    queries = read_json(attack / "tested_queries.json")
    labels = read_json(attack / "poisoned_qrels.json")
    qrels = read_qrels(dataset / "qrels/test.tsv")
    clean = read_run(base / "clean_results.run.trec")
    attack_scores = read_run(attack / "poisoned_results.run.trec")
    if not queries or set(corpus) & set(poison):
        raise ValueError("Invalid query/document sets")
    targets = {}
    for qid in queries:
        if len(labels[qid]) != 1 or qid not in qrels:
            raise ValueError(f"Invalid target labels: {qid}")
        target = next(iter(labels[qid]))
        if target not in poison or target not in attack_scores[qid] or not clean[qid]:
            raise ValueError(f"Missing inputs for query {qid}")
        targets[qid] = target
    ids = list(corpus)
    begin = time.perf_counter()
    tokens = [document_tokens(corpus[did]) for did in ids]
    tokenization_ms = (time.perf_counter() - begin) * 1000
    begin = time.perf_counter()
    clean_index = build_index(tokens)
    clean_build_ms = (time.perf_counter() - begin) * 1000
    raw, timings, diagnostic, index_build_ms = {}, {}, {}, []
    rng = random.Random(42)
    for scenario in ("clean", "attacked"):
        raw[scenario], timings[scenario], diagnostic[scenario] = {}, {}, {}
    for i, (qid, query) in enumerate(queries.items(), 1):
        target = targets[qid]
        begin = time.perf_counter()
        attacked_index = build_index(tokens + [document_tokens(poison[target])])
        index_build_ms.append((time.perf_counter() - begin) * 1000)
        for scenario, index, doc_ids in (("clean", clean_index, ids),
                                          ("attacked", attacked_index, ids + [target])):
            dense = {d: s for d, s in clean[qid].items() if d != qid}
            if scenario == "attacked":
                dense[target] = attack_scores[qid][target]
            positions = {did: pos for pos, did in enumerate(doc_ids)}

            def full_lexical():
                scores = lexical_scores(index, doc_ids, query)
                scores.pop(qid, None)
                return scores

            def subset():
                candidates = dense_shortlist(dense)
                return subset_bm25(index, positions, query, candidates)

            methods = {
                "dense": lambda: dense,
                "bm25_only": full_lexical,
                "rrf_60": lambda: reciprocal_rank_fusion(dense, full_lexical()),
                "bm25_100_then_dense": lambda: lexical_consensus(dense, full_lexical()),
                "dense_100_then_bm25": lambda: reverse_consensus(dense, subset()),
                "budgeted_rrf_100": lambda: budgeted_fusion(dense, subset()),
            }
            subset_scores = subset()
            relevant = {d for d, score in qrels[qid].items() if score > 0}
            diagnostic[scenario][qid] = {
                "dense100_has_relevant": bool(set(subset_scores) & relevant),
                "dense100_positive_bm25_count": sum(s > 0 for s in subset_scores.values()),
                "dense100_no_lexical_signal": not any(s > 0 for s in subset_scores.values()),
            }
            # One warm-up per method, then interleave five measured repetitions.
            for name, method in methods.items():
                raw[scenario].setdefault(name, {})[qid] = method()
                timings[scenario].setdefault(name, {})[qid] = []
            for _ in range(5):
                order = list(methods)
                rng.shuffle(order)
                for name in order:
                    begin = time.perf_counter()
                    scores = methods[name]()
                    sorted(scores, key=lambda d: (scores[d], d), reverse=True)[:5]
                    timings[scenario][name][qid].append((time.perf_counter() - begin) * 1000)
        if i % 10 == 0:
            print(f"Completed {i}/100 queries; elapsed {time.perf_counter()-started:.1f}s", flush=True)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_description": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "rank-bm25")},
        "settings": {"queries": len(queries), "clean_documents": len(corpus), "top_k": 5,
                     "dense_window": 100, "weighted_rrf_lexical_weight": 0.75,
                     "weighted_rrf_dense_weight": 0.25, "rrf_constant": 60,
                     "bm25_idf": "Full scenario corpus, not the 100-document candidate subset",
                     "timing_repeats_per_query": 5, "one_warmup_per_method_per_query": True,
                     "parameters_fixed_before_run": True},
        "timing_scope": "Incremental in-memory CPU scoring/ranking only. Excludes dense encoding/search, disk/network and index construction. Dense scores are reused.",
        "index_cost": {"clean_tokenization_ms": tokenization_ms, "clean_build_ms": clean_build_ms,
                       "attack_scenario_builds": stats(index_build_ms)},
        "scenarios": {}, "diagnostic": diagnostic, "timings_ms": timings,
    }
    for scenario, methods in raw.items():
        report["scenarios"][scenario] = {}
        dense_relevant = {qid for qid in queries if aggregate(
            {qid: methods["dense"][qid]}, {qid: queries[qid]}, targets, qrels, 5
        )["per_query"][qid]["clean_recall_at_k"] > 0}
        for name, scores in methods.items():
            measured = aggregate(scores, queries, targets, qrels, 5)
            good = {qid for qid, row in measured["per_query"].items() if row["clean_recall_at_k"] > 0}
            measured.update({
                "queries_with_relevant_at_5": len(good),
                "queries_without_relevant_at_5": len(queries) - len(good),
                "queries_with_fewer_than_5": sum(len(s) < 5 for s in scores.values()),
                "lost_relevant_query_vs_dense": sorted(dense_relevant - good),
                "gained_relevant_query_vs_dense": sorted(good - dense_relevant),
                "incremental_cpu_timing": stats([float(np.median(v)) for v in timings[scenario][name].values()]),
            })
            report["scenarios"][scenario][name] = measured
    expected = read_json(attack / "poisoned_metrics_poisoned.json")
    mismatch = [qid for qid, row in report["scenarios"]["attacked"]["dense"]["per_query"].items()
                if int(row["target_hit_at_k"]) != expected["recall"][qid]["Recall@5"]]
    if mismatch:
        raise ValueError(f"Baseline mismatch: {mismatch}")
    report["baseline_mismatches"] = mismatch
    report["elapsed_seconds"] = time.perf_counter() - started
    source_files = [dataset / "corpus.jsonl", dataset / "qrels/test.tsv",
                    base / "clean_results.run.trec", attack / "poisoned_results.run.trec",
                    *sorted(attack.glob("*.json")), *sorted(Path(__file__).parent.glob("*.py"))]
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    write_json(args.output / "results.json", report)
    summary = {k: v for k, v in report.items() if k not in {"scenarios", "diagnostic", "timings_ms", "source_sha256"}}
    summary["scenarios"] = {s: {m: {k: v for k, v in metrics.items() if k != "per_query"}
                                   for m, metrics in methods.items()} for s, methods in report["scenarios"].items()}
    write_json(args.output / "summary.json", summary)
    with (args.output / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "method", "query_id", "query", "target_rank", "target_hit_at_5", "recall_at_5", "ndcg_at_5", "has_relevant", "incremental_median_ms", "top5_ids"])
        for scenario, methods in report["scenarios"].items():
            for name, metrics in methods.items():
                for qid, row in metrics["per_query"].items():
                    writer.writerow([scenario, name, qid, queries[qid], row["target_rank_in_saved_pool"],
                                     row["target_hit_at_k"], row["clean_recall_at_k"], row["clean_ndcg_at_k"],
                                     row["clean_recall_at_k"] > 0, np.median(timings[scenario][name][qid]),
                                     json.dumps(row["top_ids"])])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
