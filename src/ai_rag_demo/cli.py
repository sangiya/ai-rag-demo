"""Command-line entry point for ai-rag-demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from .config import DEFAULT_CONFIG
from .ingest import build_vectorstore, default_embeddings
from .loader import load_and_split
from .qa import ask, build_qa_chain, default_llm

_SAMPLE_DOCUMENT = Path(__file__).parent / "sample_data" / "account.txt"
_DEMO_QUESTION = "What is the current account balance and when is the next billing date?"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-rag-demo",
        description="A practical LangChain + Ollama + Chroma retrieval-augmented generation demo",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="chunk and embed a document into Chroma")
    ingest.add_argument("--document", default=str(_SAMPLE_DOCUMENT))
    ingest.add_argument("--chroma-path", default="./chroma_db")
    ingest.add_argument("--collection", default=DEFAULT_CONFIG.collection_name)

    ask_cmd = subcommands.add_parser("ask", help="ask a question against an ingested collection")
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--chroma-path", default="./chroma_db")
    ask_cmd.add_argument("--collection", default=DEFAULT_CONFIG.collection_name)
    ask_cmd.add_argument("--k", type=int, default=DEFAULT_CONFIG.retrieval_k)
    ask_cmd.add_argument("--json", action="store_true")

    demo = subcommands.add_parser(
        "demo", help="run the full ingest -> retrieve -> answer pipeline on the bundled sample"
    )
    demo.add_argument("--json", action="store_true")

    serve_cmd = subcommands.add_parser(
        "serve", help="run the local web UI (requires the [web] extra)"
    )
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)

    return parser


def _print_sources(sources: list[dict[str, Any]]) -> None:
    print()
    print("Sources:")
    for index, source in enumerate(sources, start=1):
        print(f"--- Source {index} ---")
        print(source["content"][:300])


def _run_ingest(args: argparse.Namespace) -> int:
    chunks = load_and_split(args.document)
    embeddings = default_embeddings()
    build_vectorstore(
        chunks, embeddings, persist_directory=args.chroma_path, collection_name=args.collection
    )
    print(
        f"Ingested {len(chunks)} chunk(s) from {args.document} "
        f"into {args.chroma_path} (collection={args.collection})"
    )
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    embeddings = default_embeddings()
    vectorstore = Chroma(
        persist_directory=args.chroma_path,
        embedding_function=embeddings,
        collection_name=args.collection,
    )
    qa = build_qa_chain(vectorstore, default_llm(), k=args.k)
    result = ask(qa, args.question)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(result["answer"])
    _print_sources(result["sources"])
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    chunks = load_and_split(_SAMPLE_DOCUMENT)
    embeddings = default_embeddings()
    vectorstore = build_vectorstore(
        chunks, embeddings, persist_directory=None, collection_name="demo"
    )
    qa = build_qa_chain(vectorstore, default_llm())
    result = ask(qa, _DEMO_QUESTION)
    if args.json:
        print(json.dumps({"question": _DEMO_QUESTION, **result}, indent=2))
        return 0
    print(f"Question: {_DEMO_QUESTION}")
    print()
    print("Answer:", result["answer"])
    _print_sources(result["sources"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        return _run_ingest(args)

    if args.command == "ask":
        return _run_ask(args)

    if args.command == "demo":
        return _run_demo(args)

    if args.command == "serve":
        try:
            import uvicorn

            from .web import create_app
        except ImportError:
            print(
                "the web UI needs the [web] extra: pip install 'ai-rag-demo[web]'",
                file=sys.stderr,
            )
            return 1
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
