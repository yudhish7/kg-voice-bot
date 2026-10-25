"""Command-line interface: ``kgvoicebot ingest | ask | voice | serve | stats``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .factory import build_bot, build_ingestor, load_graph
from .graph import KnowledgeGraph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kgvoicebot", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build/extend the knowledge graph from .txt/.md files")
    p.add_argument("path", type=Path)
    p.add_argument("--reset", action="store_true", help="discard the existing graph first")

    p = sub.add_parser("ask", help="ask a text question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print facts and sources as JSON")

    p = sub.add_parser("voice", help="answer a spoken question from an audio file")
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("answer.wav"))

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    sub.add_parser("stats", help="show knowledge graph size")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    try:
        return _run(args, settings)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "ingest":
        kg = KnowledgeGraph() if args.reset else load_graph(settings)
        totals = build_ingestor(kg, settings).ingest_path(args.path)
        kg.save(settings.graph_path)
        print(json.dumps({**totals, **kg.stats()}))
        return 0

    kg = load_graph(settings)
    if args.command == "stats":
        print(json.dumps(kg.stats()))
        return 0

    if not kg.chunks and args.command != "serve":
        raise RuntimeError("The knowledge graph is empty. Run 'kgvoicebot ingest <path>' first.")

    if args.command == "ask":
        answer = build_bot(kg, settings, with_audio=False).ask_text(args.question)
        if args.json:
            from .api import _answer_payload

            print(json.dumps(_answer_payload(answer), indent=2))
        else:
            print(answer.text)
        return 0

    if args.command == "voice":
        reply = build_bot(kg, settings).ask_voice(args.audio.read_bytes())
        args.output.write_bytes(reply.audio)
        print(f"You said: {reply.transcript}")
        print(f"Answer:   {reply.answer.text}")
        print(f"Audio written to {args.output}")
        return 0

    if args.command == "serve":
        import uvicorn

        from .api import create_app

        bot = build_bot(kg, settings)
        app = create_app(bot, build_ingestor(kg, settings), kg, settings)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
