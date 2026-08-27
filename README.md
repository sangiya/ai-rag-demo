# ai-rag-demo

A practical retrieval-augmented generation pipeline built on the standard
industry stack: **LangChain, a local Ollama server, and a Chroma vector
store.** Where `ai-rag-engine` builds retrieval from scratch (hashing
embedder, BM25, a hand-rolled vector store) to make retrieval quality
inspectable, this repo demonstrates the other half of the skill: wiring
together the tools most RAG systems are actually built with in production.

## Why it exists

- **Uses the real ecosystem, not a reimplementation.** `OllamaEmbeddings`,
  `ChatOllama`, and `Chroma` are the same building blocks used in most
  LangChain-based RAG deployments -- knowing how to compose them is a
  distinct, practical skill from building a retrieval engine from scratch.
- **Every stage is injectable, so it's testable offline.** `build_vectorstore`
  and `build_qa_chain` take the embedder and chat model as arguments rather
  than constructing them internally, so the test suite exercises real Chroma
  similarity search and a real `RetrievalQA` chain using LangChain's
  deterministic fake embedder and fake chat model -- no network, no Ollama
  required to run `pytest`.
- **The sample document is synthetic.** `sample_data/account.txt` is a
  fabricated account summary; nothing here is drawn from real logs or
  production data.

## Architecture

```
ai_rag_demo/
├── config.py       Ollama/Chroma settings (RagConfig, DEFAULT_CONFIG)
├── loader.py           Read a document, split it into overlapping chunks
├── ingest.py               Embed chunks into a Chroma vector store
├── qa.py                       Retrieval chain + question answering
├── web.py                          Local web UI (optional [web] extra)
├── sample_data/account.txt             Bundled synthetic demo document
└── cli.py                                  Command-line entry point
```

## Install

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Runtime dependencies are the LangChain integration packages
(`langchain-core`, `langchain-text-splitters`, `langchain-chroma`,
`langchain-ollama`, `langchain-classic`) -- this repo is a glue-code demo
over that stack, not a from-scratch library. Running `ingest`, `ask`,
`demo`, or `serve` needs a local [Ollama](https://ollama.com) server with
the embedding and chat models pulled:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Add `[web]` to the install extras above (e.g. `.[dev,web]`) for the local
web UI covered below.

## Usage

### Web UI

```bash
ai-rag-demo serve
```

Opens `http://127.0.0.1:8000`: edit or accept the bundled sample document,
ingest it into an in-memory Chroma collection, then ask questions and see
the real generated answer alongside the exact chunks it was grounded in.
Needs a running local Ollama server (see above) -- this is a real pipeline,
not a canned response.

### Live demonstration

```bash
ai-rag-demo demo
```

Runs the full ingest -> retrieve -> answer pipeline against the bundled
sample document and prints a real answer with its source chunks.

### Ingest and ask against your own document

```bash
ai-rag-demo ingest --document ./my-notes.txt --chroma-path ./chroma_db --collection my-notes
ai-rag-demo ask --question "What did the notes say about pricing?" --chroma-path ./chroma_db --collection my-notes
```

### Library

```python
from ai_rag_demo import ask, build_qa_chain, build_vectorstore, default_embeddings, default_llm, load_and_split

chunks = load_and_split("my-notes.txt")
vectorstore = build_vectorstore(chunks, default_embeddings(), persist_directory="./chroma_db")
qa = build_qa_chain(vectorstore, default_llm())
result = ask(qa, "What did the notes say about pricing?")
print(result["answer"])
```

## Design notes

**The embedder and chat model are always injected, never constructed inside
`build_vectorstore` or `build_qa_chain`.** `default_embeddings()` and
`default_llm()` build the real Ollama-backed clients for CLI and web-UI use;
tests inject LangChain's `DeterministicFakeEmbedding` and `FakeListChatModel`
instead, so the same code path that runs against real Ollama in production
also runs deterministically and offline in CI.

**The document loader reads files directly rather than using
`langchain_community.TextLoader`.** `langchain-community` is being sunset
upstream; reading UTF-8 text with `pathlib` needs no deprecated dependency
for what is fundamentally reading a file.

**The web UI holds one in-memory Chroma collection per server process.**
There is no persistence beyond the running process and no multi-tenancy --
ingesting a new document replaces the previous collection, which matches
a single-user local tool rather than a hosted product.

## Development

```bash
pytest                    # 13 tests, no network required
ruff check src tests
ruff format --check src tests
mypy                      # strict
```

## License

MIT © sangiya
