# kgvoicebot

An end-to-end **voice service bot** built on **Knowledge-Graph RAG**: a caller's speech is transcribed, the question is answered from a knowledge graph (plus the source passages it was built from), and the answer is spoken back.

```text
 audio ─▶ ASR ─▶ question ─▶ ┌──────────────── GraphRetriever ────────────────┐ ─▶ evidence ─▶ Generator ─▶ answer ─▶ TTS ─▶ audio
                             │ 1 link entities in the question                │                 (LLM or
                             │ 2 expand N hops through the knowledge graph    │                  extractive)
                             │ 3 rank chunks: BM25 blended with graph distance│
                             │ 4 select nearby facts (triples)                │
                             └────────────────────────────────────────────────┘
```

## Why a knowledge graph?

Plain vector/keyword RAG retrieves passages that *look like* the question. Banking questions are often **multi-hop** – *"What is the ATM limit on the card that comes with Premium Savings?"* needs `Premium Savings account → includes → Platinum debit card` **and** `Platinum debit card → has → daily ATM limit of 100000 rupees`, which may live in different paragraphs. Walking the graph pulls both facts in together, and the facts are handed to the generator next to the original passages, so answers stay grounded and traceable.

## Features

- **Ingestion**: `.txt` / `.md` → sentence-aware chunks → triples → graph. Re-ingesting a document replaces it (idempotent).
- **Pluggable extraction**: offline rule-based baseline, or LLM-based triple extraction (Claude).
- **Hybrid retrieval**: entity linking (exact + fuzzy), *N*-hop expansion, BM25 blended with graph proximity, graceful fallback to BM25 when no entity is recognised.
- **Grounded answers**: the model only sees retrieved context; with no evidence the bot says it doesn't know *without calling the model*.
- **Pluggable audio**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for ASR, `espeak-ng` for offline TTS, plus test stubs.
- **Interfaces**: CLI and an HTTP API (`/ask`, `/voice`, `/ingest`, `/health`) with optional bearer-token auth.
- **Runs fully offline** (rules + extractive generator + stubs) – no keys needed to try it.

## Quick start

```bash
git clone <this repo> && cd kg-voice-bot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests

# Offline demo: no API key, no speech engine
export KGV_EXTRACTOR=rules KGV_GENERATOR=extractive KGV_TTS=stub KGV_ASR=text
kgvoicebot ingest data/sample --reset
kgvoicebot ask "What credit score does the Home Loan need?"
# -> The Home Loan requires a credit score of 700.
kgvoicebot ask --json "What is the ATM limit on the card that comes with Premium Savings?"
# -> JSON with the graph facts (Premium Savings -> Platinum debit card -> ATM limit 100000 rupees)
#    and source chunks. The offline extractive generator quotes sentences, so for a fluent
#    multi-hop answer use the LLM generator (set ANTHROPIC_API_KEY).
```

### Full stack (real speech + LLM)

```bash
pip install -e ".[llm,whisper]"
sudo apt install espeak-ng ffmpeg          # TTS engine, audio decoding for Whisper
export ANTHROPIC_API_KEY=...               # enables LLM extraction + answers (auto-detected)

kgvoicebot ingest data/sample --reset
kgvoicebot voice question.wav -o answer.wav   # speech in, speech out
kgvoicebot serve --port 8000
```

### HTTP API

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ask  -H 'content-type: application/json' \
     -d '{"question": "How do I block a lost card?"}'
curl -s localhost:8000/voice --data-binary @question.wav -o answer.wav -D -   # transcript in X-Transcript
curl -s localhost:8000/ingest -H 'content-type: application/json' \
     -d '{"doc_id": "faq", "text": "The Gold plan includes a smart speaker."}'
```

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | – | `{"status","chunks","entities","relations"}` (never needs auth) |
| `POST /ask` | `{"question"}` (≤ 1 000 chars) | `{"answer","entities","facts":[…],"sources":[chunk ids]}` |
| `POST /voice` | raw audio bytes (≤ `KGV_MAX_AUDIO_BYTES`) | `audio/wav`; URL-encoded `X-Transcript` and `X-Answer` headers |
| `POST /ingest` | `{"doc_id","text"}` | counts for the document + graph totals; graph is persisted |

Errors are JSON `{"error": "..."}` with 400 (bad JSON / empty audio), 401, 413 (too large), 422 (bad field). Set `KGV_API_KEY` to require `Authorization: Bearer <key>` on everything except `/health`. The service has no TLS or rate limiting – put it behind a reverse proxy for anything public.

### Docker

```bash
docker build -t kgvoicebot .
docker run -p 8000:8000 -v kg-data:/data -e ANTHROPIC_API_KEY kgvoicebot
```

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)).

| Variable | Default | Meaning |
|---|---|---|
| `KGV_DATA_DIR` | `.kgvoicebot` | where `graph.json` is stored |
| `KGV_EXTRACTOR` | `auto` | `rules`, `llm`, or `auto` (LLM if `ANTHROPIC_API_KEY` is set and the SDK is installed) |
| `KGV_GENERATOR` | `auto` | `llm`, `extractive`, or `auto` (same rule) |
| `KGV_ASR` | `auto` | `whisper` (`auto` = whisper) or `text` (treat bytes as UTF-8 text; tests/demos) |
| `KGV_TTS` | `auto` | `espeak`, `stub` (tones), or `auto` (espeak if installed, otherwise stub with a warning) |
| `KGV_LLM_MODEL` | `claude-sonnet-5-5` | model used for extraction and answers |
| `KGV_WHISPER_MODEL` | `base` | faster-whisper model size |
| `KGV_HOPS` / `KGV_TOP_K` / `KGV_MAX_FACTS` | `2` / `4` / `15` | retrieval depth, chunks returned, facts returned |
| `KGV_MAX_AUDIO_BYTES` | `10485760` | upload limit for `/voice` |
| `KGV_API_KEY` | unset | enables bearer-token auth |

## Project layout

```text
src/kgvoicebot/
  text.py chunker.py      text utilities, sentence-aware chunking
  extract.py              TripleExtractor: RuleBasedExtractor, LLMExtractor
  graph.py                KnowledgeGraph (networkx MultiDiGraph + provenance + JSON persistence)
  bm25.py retriever.py    lexical index; graph-aware hybrid retrieval
  generator.py llm.py     Generator: LLMGenerator, ExtractiveGenerator; Anthropic client
  audio/asr.py tts.py     ASR / TTS backends
  bot.py                  VoiceBot: audio -> text -> evidence -> answer -> audio
  api.py cli.py           Starlette app; argparse CLI
  config.py factory.py    env-based settings; component wiring
tests/                    60 tests (unit, retrieval, bot, live HTTP server, CLI)
data/sample/              small fictional "Acme Bank" help centre (accounts, cards, loans, support) used by the demo and tests
```

Every external dependency (ASR, TTS, LLM, extractor, generator) sits behind a small `Protocol`, so swapping a backend means writing one class and one line in `factory.py`.

## Design decisions

- **Graph as an index, not the source of truth.** Every node and edge records the chunks it came from, so retrieval returns both structured facts and original text, and an extraction mistake can't silently invent content.
- **Graph proximity is a soft signal.** Chunks are ranked by `(1-w)·BM25 + w·1/(1+distance)` (`w=0.5`). Without recognised entities `w=0` and it is plain BM25, so unknown phrasing still works.
- **Refuse before generating.** Empty evidence short-circuits to "I don't have information about that"; the LLM prompt also instructs it to say so when context is insufficient.
- **Atomic persistence.** The graph is written to a temp file then renamed, so a crash never leaves a corrupt `graph.json`.
- **Concurrency.** A re-entrant lock guards graph mutation and retrieval; slow LLM extraction runs *outside* the lock so ingestion doesn't block queries.
- **Lean core.** Runtime deps are `networkx`, `starlette`, `uvicorn`; heavy ones (`anthropic`, `faster-whisper`) are optional extras imported lazily.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.10–3.13.

## Limitations and next steps

Being upfront about what this is and isn't:

- **Rule-based extraction is a baseline.** It only understands short declarative sentences matching a fixed set of verbs (see `extract.py`). For real documents use the LLM extractor (`KGV_EXTRACTOR=llm`).
- **The extractive generator can't tell when retrieved text doesn't answer the question** – it returns the closest sentences. It exists for offline use and tests; use the LLM generator in production. Generic words ("plan", "include") shared with unrelated passages can surface weak matches.
- **No entity resolution beyond normalisation and fuzzy word overlap** – "Premium account" and "Premium Savings account" are different nodes. Embedding-based linking and coreference would be the next step.
- **No conversation memory** – each question is independent; follow-ups like "and how much is it?" need session state.
- **Batch audio, not streaming** – `/voice` takes a complete recording. Streaming ASR/TTS and barge-in would be needed for a phone-grade agent.
- **Single process, in-memory graph** – fine for thousands of facts; a graph database would be the move at larger scale.
- **Verified here:** the retrieval, bot, HTTP API and CLI paths (60 tests) using the offline backends. `faster-whisper`, `espeak-ng`, the Anthropic client and the Docker image need their own dependencies/network and were **not** exercised in the build environment – their wrappers are intentionally thin, but smoke-test them (`kgvoicebot voice …`) before relying on them.

## License

MIT
