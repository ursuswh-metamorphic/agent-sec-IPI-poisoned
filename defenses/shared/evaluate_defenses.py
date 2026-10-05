"""Replay Kaggle retrieval scores with two gates. No GPU or model required."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from defenses import content_hash, detect_injection, filter_results, trust_reasons


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_corpus(path):
    documents = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            if doc_id in documents:
                raise ValueError(f"Duplicate corpus id: {doc_id}")
            documents[doc_id] = doc
    return documents


def read_run(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            qid, _, did, _, score, _ = line.split()
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("Non-finite retrieval score")
            query = result.setdefault(qid, {})
            if did in query:
                raise ValueError(f"Duplicate run entry: {qid}/{did}")
            query[did] = value
    return result


def read_qrels(path):
    result = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return result


def ranking(scores, qid):
    # BEIR excludes identical query/document IDs. TREC breaks score ties by ID.
    return sorted((did for did in scores if did != qid),
                  key=lambda did: (scores[did], did), reverse=True)


def utility(top, labels, k):
    relevant = {did for did, value in labels.items() if value > 0}
    recall = len(set(top) & relevant) / len(relevant) if relevant else 0.0
    dcg = sum(labels.get(did, 0) / math.log2(i + 2) for i, did in enumerate(top))
    ideal = sum(value / math.log2(i + 2) for i, value in enumerate(
        sorted((value for value in labels.values() if value > 0), reverse=True)[:k]))
    return recall, dcg / ideal if ideal else 0.0


def aggregate(results, queries, targets, qrels, k):
    rows = {}
    for qid in queries:
        ordered = ranking(results[qid], qid)
        target = targets[qid]
        recall, ndcg = utility(ordered[:k], qrels[qid], k)
        rows[qid] = {
            "target_id": target,
            "target_rank_in_saved_pool": ordered.index(target) + 1 if target in ordered else None,
            "target_hit_at_k": target in ordered[:k],
            "clean_recall_at_k": recall,
            "clean_ndcg_at_k": ndcg,
            "top_ids": ordered[:k],
        }
    count = len(rows)
    return {
        "queries": count,
        "target_hits_at_k": sum(row["target_hit_at_k"] for row in rows.values()),
        "target_hit_rate_at_k": sum(row["target_hit_at_k"] for row in rows.values()) / count,
        "clean_recall_at_k": sum(row["clean_recall_at_k"] for row in rows.values()) / count,
        "clean_ndcg_at_k": sum(row["clean_ndcg_at_k"] for row in rows.values()) / count,
        "per_query": rows,
    }


def enroll(args):
    corpus = read_corpus(args.corpus)
    write_json(args.registry, {
        "schema": 1,
        "assumption": "Operator treats the original benchmark corpus as approved before ingestion of attack documents.",
        "source": str(args.corpus),
        "documents": {did: {"sha256": content_hash(doc), "source": "approved_clean_corpus_registry"}
                      for did, doc in corpus.items()},
    })
    print(f"Enrolled {len(corpus)} benchmark documents: {args.registry}")


def replay(args):
    base, attack = args.base, args.attack
    corpus = read_corpus(args.corpus)
    poison = read_json(attack / "poisoned_added_corpus.json")
    queries = read_json(attack / "tested_queries.json")
    target_labels = read_json(attack / "poisoned_qrels.json")
    qrels = read_qrels(args.qrels)
    clean_run = read_run(base / "clean_results.run.trec")
    poison_run = read_run(attack / "poisoned_results.run.trec")
    registry = read_json(args.registry)["documents"]
    if not queries or set(poison) & set(corpus):
        raise ValueError("Empty evaluation or colliding document IDs")
    targets, combined = {}, {}
    for qid in queries:
        labels = target_labels[qid]
        if len(labels) != 1 or qid not in qrels:
            raise ValueError(f"Invalid query labels: {qid}")
        target = next(iter(labels))
        if target not in poison or target not in poison_run[qid] or not clean_run[qid]:
            raise ValueError(f"Missing scores/documents for {qid}")
        targets[qid] = target
        combined[qid] = {**clean_run[qid], target: poison_run[qid][target]}
    if set(targets.values()) != set(poison):
        raise ValueError("Attack document count does not match evaluated queries")
    all_docs = {**corpus, **poison}
    report = {
        "experiment": "Offline gate replay over saved Kaggle scores; one target document per query",
        "k": args.k,
        "clean_document_count": len(corpus),
        "attack_document_count": len(poison),
        "clean_run_query_count": len(clean_run),
        "tested_query_count": len(queries),
        "clean_pool_sizes": sorted({len(clean_run[qid]) for qid in queries}),
        "clean_baseline": aggregate(clean_run, queries, targets, qrels, args.k),
        "policies": {},
        "limitations": [
            "No LLM/tool execution or adaptive attack rerun.",
            "The saved payload is a harmless retrieval test marker.",
            "Trusted-only assumes all original benchmark documents were approved; trust is not inferred from text.",
            "Replay retains the original scores and title handling, including any original scoring limitations.",
            "Refill uses the saved candidate pool, not the full corpus beyond that pool.",
        ],
    }
    for policy in ("none", "rules", "trusted_only"):
        filtered, blocked = filter_results(combined, all_docs, policy, registry)
        measured = aggregate(filtered, queries, targets, qrels, args.k)
        classify = (lambda did, doc: []) if policy == "none" else (
            (lambda did, doc: detect_injection(doc)) if policy == "rules" else
            (lambda did, doc: trust_reasons(did, doc, registry)))
        blocked_clean = {did: classify(did, doc) for did, doc in corpus.items() if classify(did, doc)}
        blocked_poison = {did: classify(did, doc) for did, doc in poison.items() if classify(did, doc)}
        measured.update({
            "blocked_clean_documents": len(blocked_clean),
            "clean_block_rate": len(blocked_clean) / len(corpus),
            "blocked_attack_documents": len(blocked_poison),
            "attack_block_rate": len(blocked_poison) / len(poison),
        })
        report["policies"][policy] = measured
        write_json(args.output / f"{policy}_blocked.json", {
            "clean_corpus": blocked_clean, "attack_corpus": blocked_poison,
            "retrieved_pool": blocked,
        })
        write_json(args.output / f"{policy}_top_k.json", {
            qid: ranking(scores, qid)[:args.k] for qid, scores in filtered.items()
        })
    # Validate replay against the author's exported metrics for the same queries.
    exported = read_json(attack / "poisoned_metrics_poisoned.json")
    differences = []
    for qid, row in report["policies"]["none"]["per_query"].items():
        expected = exported["recall"][qid][f"Recall@{args.k}"]
        if abs(expected - float(row["target_hit_at_k"])) > 1e-5:
            differences.append(qid)
    report["exported_attack_recall_mismatches"] = differences
    report["exported_metric_query_counts"] = {
        name: {metric: len(values) for metric, values in read_json(attack / name).items()}
        for name in ("poisoned_metrics_poisoned.json", "poisoned_metrics_clean.json", "clean_metrics_clean.json")
    }
    report["clean_metric_max_abs_difference"] = {}
    for filename, measured in (("clean_metrics_clean.json", report["clean_baseline"]),
                              ("poisoned_metrics_clean.json", report["policies"]["none"])):
        original = read_json(attack / filename)
        report["clean_metric_max_abs_difference"][filename] = max(
            abs(row[key] - original[group][qid][f"{label}@{args.k}"])
            for qid, row in measured["per_query"].items()
            for key, group, label in (("clean_recall_at_k", "recall", "Recall"),
                                      ("clean_ndcg_at_k", "ndcg", "NDCG")))
    source_files = [args.corpus, args.qrels, args.registry, base / "clean_results.run.trec",
                    *sorted(attack.glob("*.json")), attack / "poisoned_results.run.trec",
                    Path(__file__), Path(__file__).with_name("defenses.py")]
    report["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    write_json(args.output / "results.json", report)
    compact = {key: value for key, value in report.items()
               if key not in {"policies", "clean_baseline", "input_sha256"}}
    compact["clean_baseline"] = {key: value for key, value in report["clean_baseline"].items() if key != "per_query"}
    compact["policies"] = {name: {key: value for key, value in values.items() if key != "per_query"}
                           for name, values in report["policies"].items()}
    write_json(args.output / "summary.json", compact)
    print(json.dumps(compact, indent=2))
    if differences:
        raise ValueError("Replay differs from exported attack recall; inspect results before reporting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    enrollment = commands.add_parser("enroll-benchmark")
    enrollment.add_argument("--corpus", type=Path, required=True)
    enrollment.add_argument("--registry", type=Path, required=True)
    enrollment.set_defaults(func=enroll)
    evaluation = commands.add_parser("replay")
    for name in ("base", "attack", "corpus", "qrels", "registry", "output"):
        evaluation.add_argument(f"--{name}", type=Path, required=True)
    evaluation.add_argument("--k", type=int, choices=(1, 3, 5, 10, 100, 1000), default=5)
    evaluation.set_defaults(func=replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
