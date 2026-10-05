"""Retrieval gates. Trust must come from a separately protected registry."""

import hashlib
import json
import re
import unicodedata


RULES = {
    "override_instructions": r"\b(?:ignore|disregard|override)\b.{0,60}\b(?:instructions?|rules?|system prompt)\b",
    "role_spoofing": r"(?:<\|(?:im_start|start_header_id)\|>\s*(?:system|developer)|\[system\]|\bsystem\s*(?:message|override)\s*:)",
    "secret_access": r"(?:\.env\b|~[/\\]\.ssh\b|[/\\]\.ssh[/\\]|\b(?:reveal|send|print|exfiltrate)\b.{0,60}\b(?:credentials?|api[ _-]?keys?|passwords?|secrets?)\b)",
    "tool_directive": r"\b(?:execute|run|invoke|call)\s+(?:the\s+)?(?:shell|bash|powershell|execute_bash_code|execute_python_code|send_email)\b",
}
COMPILED = {name: re.compile(pattern, re.I | re.S) for name, pattern in RULES.items()}


def document_text(document):
    return str(document.get("title", "")) + "\n" + str(document.get("text", ""))


def detect_injection(document):
    text = unicodedata.normalize("NFKC", document_text(document))
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return [name for name, pattern in COMPILED.items() if pattern.search(text)]


def content_hash(document):
    content = {key: str(document.get(key, "")) for key in ("title", "text")}
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def trust_reasons(doc_id, document, registry):
    entry = registry.get(str(doc_id))
    if entry is None:
        return ["not_enrolled"]
    if entry.get("sha256") != content_hash(document):
        return ["content_changed"]
    return []


def filter_results(results, corpus, policy="rules", registry=None):
    """Filter a ranked candidate pool before selecting top-k; fail on missing text."""
    if policy not in {"none", "rules", "trusted_only"}:
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "trusted_only" and registry is None:
        raise ValueError("trusted_only requires a protected registry")
    filtered, blocked, decisions = {}, {}, {}
    for query_id, scores in results.items():
        filtered[query_id] = {}
        for doc_id, score in scores.items():
            if doc_id not in corpus:
                raise ValueError(f"Missing document: {doc_id}")
            if doc_id not in decisions:
                reasons = []
                if policy == "rules":
                    reasons = detect_injection(corpus[doc_id])
                elif policy == "trusted_only":
                    reasons = trust_reasons(doc_id, corpus[doc_id], registry)
                decisions[doc_id] = reasons
            reasons = decisions[doc_id]
            if reasons:
                blocked[doc_id] = reasons
            else:
                filtered[query_id][doc_id] = score
    return filtered, blocked
