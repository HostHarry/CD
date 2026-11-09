import argparse
import json
from .config import resolve_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cdapc")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "verify-native", "verify-adapter", "infer"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        if name == "infer":
            item.add_argument("--resume-run")
            item.add_argument("--max-samples", type=int, help="Interrupt after N new samples for resume acceptance")
    for name in ("score", "summarize", "replay"):
        item = sub.add_parser(name)
        item.add_argument("--run", required=True)
        if name == "score":
            item.add_argument("--profile", default="official_and_strict")
        elif name == "summarize":
            item.add_argument("--evaluation")
        else:
            item.add_argument("--event", default="first_eligible_flip")
            item.add_argument("--design", default="content_swap")
    item = sub.add_parser("pope-report")
    item.add_argument("--runs", nargs="*", default=[])
    item.add_argument("--evaluations", nargs="*")
    item.add_argument("--native-export")
    item.add_argument("--allow-partial", action="store_true")
    item.add_argument("--output", required=True)
    item = sub.add_parser("mmmu-judge")
    item.add_argument("--run", required=True)
    item.add_argument("--judge", default="deepseek-flash")
    item = sub.add_parser("mmmu-report")
    item.add_argument("--runs", nargs="*", default=[])
    item.add_argument("--evaluations", nargs="*")
    item.add_argument("--native-runs", nargs="*")
    item.add_argument("--native-evaluations", nargs="*")
    item.add_argument("--judge", default="deepseek-flash")
    item.add_argument("--allow-partial", action="store_true")
    item.add_argument("--output", required=True)
    item = sub.add_parser("compare")
    for name in ("run-a", "run-b", "evaluation-a", "evaluation-b"):
        item.add_argument("--" + name, required=True)
    item.add_argument("--metric", default="strict_accuracy")
    item.add_argument("--unit", default="sample_id")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "config"):
            config = resolve_config(args.config)
        if args.command == "doctor":
            from .experiments.infer import preflight
            samples, manifest, provenance = preflight(config)
            result = {"passed": True, "sample_count": len(samples), "sample_ids": [s.sample_id for s in samples], **provenance}
        elif args.command.startswith("verify-"):
            from .experiments.verify import verify
            result = verify(config, args.command.split("-")[1])
            if not result["passed"]:
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
        elif args.command == "infer":
            from .experiments.infer import infer
            if args.max_samples is not None and args.max_samples < 1:
                raise ValueError("max-samples must be positive")
            result = {"run": str(infer(config, args.resume_run, max_samples=args.max_samples))}
        elif args.command == "score":
            from .experiments.score import score_run
            result = score_run(args.run, args.profile)
        elif args.command == "summarize":
            from .experiments.score import summarize
            result = summarize(args.run, args.evaluation)
        elif args.command == "pope-report":
            from .experiments.pope_report import pope_report
            result = pope_report(args.runs, args.evaluations, args.native_export, args.allow_partial, args.output)
        elif args.command == "mmmu-judge":
            from .evaluation.llm_judge import mmmu_judge
            result = mmmu_judge(args.run, args.judge)
        elif args.command == "mmmu-report":
            from .experiments.mmmu_report import mmmu_report
            result = mmmu_report(args.runs, args.evaluations, args.native_runs,
                                 args.native_evaluations, args.judge, args.allow_partial, args.output)
        elif args.command == "compare":
            from .analysis.paired import compare
            result = compare(args.run_a, args.evaluation_a, args.run_b, args.evaluation_b, args.metric, args.unit)
        elif args.command == "replay":
            from .experiments.replay import replay
            result = replay(args.run, args.event, args.design)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
