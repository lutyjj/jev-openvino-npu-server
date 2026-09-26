import math

import numpy as np

from jev_server.api.mapping import question_keys


def distributions(case, answers):
    if set(answers) != set(case.request.questions):
        raise ValueError("Answer IDs differ from question IDs")
    result = {}
    for name, question in case.request.questions.items():
        answer = answers[name]
        if answer["type"] != question.type:
            raise ValueError("Answer type differs from question type")
        values = (
            {"false": 1 - answer["noul"], "true": answer["noul"]}
            if question.type == "noul"
            else answer["probabilities"]
        )
        keys = question_keys(question.type, question.criteria)
        if set(keys) != set(values):
            raise ValueError("Probability keys differ from options")
        p = np.array([values[key] for key in keys], dtype=float)
        if (
            not np.isfinite(p).all()
            or (p < 0).any()
            or (p > 1).any()
            or abs(p.sum() - 1) > 0.02
        ):
            raise ValueError("Invalid probability distribution")
        result[name] = dict(zip(keys, (p / p.sum()).tolist()))
    return result


def percentiles(values):
    return (
        {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        if values
        else None
    )


def summarize(rows):
    samples = [sample for row in rows for sample in row["samples"]]
    scored = [
        (row["expected"][name], p)
        for row in rows
        if row["samples"]
        for name, p in row["samples"][0]["probabilities"].items()
    ]
    return {
        "cases": len(rows),
        "failed_cases": sum(bool(row["error"]) for row in rows),
        "measured_requests": len(samples),
        "request_ms": percentiles([s["elapsed_ms"] for s in samples]),
        "answer_ms": percentiles(
            [s["answer_ms"] for s in samples if s["answer_ms"] is not None]
        ),
        "questions_per_second": sum(len(s["probabilities"]) for s in samples)
        / (sum(s["elapsed_ms"] for s in samples) / 1000)
        if samples
        else None,
        "quality": {
            "sample_policy": "First measured repeat per question; categorical mode for score questions",
            "questions": len(scored),
            "accuracy": sum(max(p, key=p.get) == y for y, p in scored) / len(scored)
            if scored
            else None,
            "brier": float(
                np.mean(
                    [
                        sum((value - (key == y)) ** 2 for key, value in p.items())
                        for y, p in scored
                    ]
                )
            )
            if scored
            else None,
            "nll": float(np.mean([-math.log(max(p[y], 1e-12)) for y, p in scored]))
            if scored
            else None,
        },
    }


def compare(baseline, candidate):
    for key in ("format_version", "corpus_sha256", "settings"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare different {key}")
    if baseline["target"]["mode"] != candidate["target"]["mode"]:
        raise ValueError("Cannot compare HTTP timing with runtime timing")
    if not baseline["target"].get("artifacts") or not candidate["target"].get(
        "artifacts"
    ):
        raise ValueError(
            "Both targets must report loaded artifact identities; update older servers"
        )
    if baseline["target"].get("execution_devices") != candidate["target"].get(
        "execution_devices"
    ):
        raise ValueError("Cannot compare different execution devices")
    left, right = baseline["cases"], candidate["cases"]
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Case sets differ")
    deltas, agreement, ratios = [], [], []
    for a, b in zip(left, right):
        if (
            a["error"]
            or b["error"]
            or len(a["samples"]) != baseline["settings"]["repeats"]
            or len(b["samples"]) != candidate["settings"]["repeats"]
        ):
            raise ValueError("Both runs must complete every case and repeat")
        ratios.append(
            np.median([s["elapsed_ms"] for s in a["samples"]])
            / np.median([s["elapsed_ms"] for s in b["samples"]])
        )
        for sa, sb in zip(a["samples"], b["samples"]):
            if set(sa["probabilities"]) != set(sb["probabilities"]):
                raise ValueError("Question sets differ")
            for name, p in sa["probabilities"].items():
                q = sb["probabilities"][name]
                if set(p) != set(q):
                    raise ValueError("Option sets differ")
                agreement.append(max(p, key=p.get) == max(q, key=q.get))
                deltas.append(max(abs(p[k] - q[k]) for k in p))
    return {
        "paired_case_speedup": percentiles(ratios),
        "decision_agreement": float(np.mean(agreement)),
        "max_probability_delta": max(deltas),
        "mean_max_probability_delta": float(np.mean(deltas)),
        "accuracy_delta": candidate["summary"]["quality"]["accuracy"]
        - baseline["summary"]["quality"]["accuracy"],
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
    }
