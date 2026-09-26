import argparse
import json
from pathlib import Path

from jev_server.bench.corpus import import_kev, load_corpus, synthetic, write_corpus
from jev_server.bench.metrics import compare
from jev_server.bench.runner import run
from jev_server.bench.targets import HttpTarget, RuntimeTarget


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--count", type=int, default=256)
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument(
        "--profile", choices=["small", "context2048"], default="small"
    )
    generate.add_argument("--output", type=Path, required=True)
    external = commands.add_parser("import-kev")
    external.add_argument("--input", type=Path, required=True)
    external.add_argument("--sha256", required=True)
    external.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    target = execute.add_mutually_exclusive_group(required=True)
    target.add_argument("--model-dir", type=Path)
    target.add_argument("--url")
    execute.add_argument("--model", default="jev-latest")
    execute.add_argument("--device", choices=["CPU", "NPU"], default="NPU")
    execute.add_argument("--corpus", type=Path, required=True)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--warmup", type=int, default=1)
    execute.add_argument("--seed", type=int, default=17)
    execute.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        if args.count < 1:
            parser.error("count must be positive")
        write_corpus(args.output, synthetic(args.count, args.seed, args.profile))
    elif args.command == "import-kev":
        write_corpus(args.output, import_kev(args.input, args.sha256))
    elif args.command == "compare":
        print(
            json.dumps(
                compare(
                    json.loads(args.baseline.read_text()),
                    json.loads(args.candidate.read_text()),
                ),
                indent=2,
                allow_nan=False,
            )
        )
    else:
        if args.repeats < 1 or args.warmup < 0:
            parser.error("repeats must be positive and warmup nonnegative")
        if args.output.exists():
            parser.error("output already exists")
        cases, digest = load_corpus(args.corpus)
        target = (
            RuntimeTarget(args.model_dir, args.device)
            if args.model_dir
            else HttpTarget(args.url, args.model, args.device)
        )
        report = run(target, cases, digest, args.repeats, args.warmup, args.seed)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps(report["summary"], indent=2, allow_nan=False))
        if report["summary"]["failed_cases"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
