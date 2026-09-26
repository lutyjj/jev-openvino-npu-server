import hashlib
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from jev_server.bench.metrics import distributions, summarize


def run(target, cases, corpus_sha256, repeats=3, warmup=1, seed=17):
    if repeats < 1 or warmup < 0:
        raise ValueError("Repeats must be positive and warmup nonnegative")
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    started_at = datetime.now(UTC).isoformat()
    rows = {
        case.id: {
            "id": case.id,
            "group": case.group,
            "expected": case.expected,
            "samples": [],
            "error": None,
        }
        for case in cases
    }
    rng = random.Random(seed)
    for iteration in range(-warmup, repeats):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            row = rows[case.id]
            if row["error"]:
                continue
            try:
                start = time.perf_counter()
                answers, answer_ms, tokens = target.predict(case)
                elapsed_ms = (time.perf_counter() - start) * 1000
                probabilities = distributions(case, answers)
                if iteration >= 0:
                    row["samples"].append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "answer_ms": answer_ms,
                            "input_tokens": tokens,
                            "probabilities": probabilities,
                        }
                    )
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                row["error"] = {"type": type(error).__name__, "message": str(error)}
        print(
            f"Completed {'warmup' if iteration < 0 else 'measured'} pass {iteration + warmup + 1}/{warmup + repeats}",
            flush=True,
        )
    values = list(rows.values())
    return {
        "format_version": 1,
        "started_at": started_at,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "corpus_sha256": corpus_sha256,
        "source_data_sha256": sorted(
            {case.source_sha256 for case in cases if case.source_sha256}
        ),
        "settings": {"repeats": repeats, "warmup": warmup, "seed": seed},
        "target": target.evidence,
        "memory": target.memory(),
        "summary": summarize(values),
        "groups": {
            group: summarize([row for row in values if row["group"] == group])
            for group in sorted({row["group"] for row in values})
        },
        "cases": values,
    }
