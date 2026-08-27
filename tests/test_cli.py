from ai_rag_demo.cli import _build_parser


class TestParser:
    def test_ingest_defaults_to_the_bundled_sample_document(self) -> None:
        args = _build_parser().parse_args(["ingest"])
        assert args.document.endswith("account.txt")
        assert args.chroma_path == "./chroma_db"

    def test_ask_requires_a_question(self) -> None:
        args = _build_parser().parse_args(["ask", "--question", "What is the balance?"])
        assert args.question == "What is the balance?"
        assert args.k == 3

    def test_demo_and_serve_are_registered(self) -> None:
        demo_args = _build_parser().parse_args(["demo"])
        assert demo_args.command == "demo"
        serve_args = _build_parser().parse_args(["serve", "--port", "9000"])
        assert serve_args.command == "serve"
        assert serve_args.port == 9000
