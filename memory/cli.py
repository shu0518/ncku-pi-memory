"""給 Pi bridge 呼叫的 CLI：capture / retrieve / inject。學生不需修改。"""
from __future__ import annotations
import argparse
import json
import sys

from .core import capture, retrieve, build_injection, make_observation


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m memory.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--summary", required=True)
    c.add_argument("--session", default="cli")
    c.add_argument("--tags", default="")

    r = sub.add_parser("retrieve")
    r.add_argument("--query", required=True)
    r.add_argument("--k", type=int, default=8)

    i = sub.add_parser("inject")
    i.add_argument("--query", required=True)
    i.add_argument("--budget", type=int, default=2000)

    args = p.parse_args(argv)
    if args.cmd == "capture":
        tags = [t for t in args.tags.split(",") if t]
        capture(make_observation(args.summary, session_id=args.session, tags=tags))
        print(f"Remembered: {args.summary}")
    elif args.cmd == "retrieve":
        print(json.dumps(retrieve(args.query, args.k), ensure_ascii=False))
    elif args.cmd == "inject":
        sys.stdout.write(build_injection(args.query, args.budget))


if __name__ == "__main__":
    main()
