"""Explicit single-job execution; committed protocol and no automatic retries."""
import argparse
import json
from core import HERE, validate
from training import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "pilot", "confirmation"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = json.loads((HERE / f"jobs_{args.phase}.json").read_text())
    if args.index is not None and not 0 <= args.index < len(jobs):
        parser.error("job index outside registered matrix")
    if not args.execute:
        print(json.dumps(jobs[args.index] if args.index is not None else
                         dict(phase=args.phase, jobs=len(jobs), executed=False), indent=1))
        return
    if args.index is None:
        parser.error("--execute requires --index")
    job = jobs[args.index]
    validate(job)
    out = HERE / "runs" / args.phase / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fit_attempt.json").open("x") as stream:
        json.dump(dict(job=job), stream)
    fit(job)


if __name__ == "__main__":
    main()
