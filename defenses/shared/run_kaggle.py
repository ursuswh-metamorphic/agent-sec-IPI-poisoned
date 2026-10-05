"""Run the documented SciFact replay against an existing Kaggle output tree."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/defense_results"))
    parser.add_argument("--approve-clean-benchmark", action="store_true", required=True,
                        help="Explicitly assume the original benchmark corpus was approved")
    args = parser.parse_args()
    base = args.root / "usenix_1226_AE/results/scifact/gte_modernbert_base"
    attack = base / "poisoned/10_5000_30_0.2_0.55_prefix_42"
    dataset = args.root / "usenix_1226_AE_work_gte/datasets/scifact"
    corpus = dataset / "corpus.jsonl"
    qrels = dataset / "qrels/test.tsv"
    needed = [corpus, qrels, base / "clean_results.run.trec"] + [attack / name for name in (
        "poisoned_added_corpus.json", "tested_queries.json", "poisoned_qrels.json",
        "poisoned_results.run.trec", "poisoned_metrics_poisoned.json",
        "poisoned_metrics_clean.json", "clean_metrics_clean.json")]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        parser.error("Missing inputs:\n" + "\n".join(missing))
    if (args.output / "results.json").exists():
        parser.error("Output already has results.json; choose a new output directory")
    registry = args.output / "trusted_registry.json"
    executable = [sys.executable, str(Path(__file__).with_name("evaluate_defenses.py"))]
    subprocess.run(executable + ["enroll-benchmark", "--corpus", str(corpus),
                                "--registry", str(registry)], check=True)
    subprocess.run(executable + ["replay", "--base", str(base), "--attack", str(attack),
                                "--corpus", str(corpus), "--qrels", str(qrels),
                                "--registry", str(registry), "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
