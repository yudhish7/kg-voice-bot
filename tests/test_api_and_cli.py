"""API tests run a real uvicorn server on a free port and talk to it with urllib,
so they need no test-only dependencies."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote

import pytest
import uvicorn

from kgvoicebot.api import create_app
from kgvoicebot.cli import main
from kgvoicebot.config import Settings
from kgvoicebot.extract import RuleBasedExtractor
from kgvoicebot.ingest import Ingestor

from .conftest import SAMPLE_DIR


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _lower(headers) -> dict[str, str]:
    # HTTP header names are case-insensitive; servers may send any casing.
    return {k.lower(): v for k, v in headers.items()}


class Server:
    def __init__(self, bot, kg, settings: Settings) -> None:
        app = create_app(bot, Ingestor(kg, RuleBasedExtractor()), kg, settings)
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> Server:
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def request(self, method: str, path: str, body: bytes | None = None, headers=None):
        req = urllib.request.Request(
            self.base + path, data=body, method=method, headers=headers or {}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, _lower(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, _lower(err.headers), err.read()

    def post_json(self, path: str, payload, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        return self.request("POST", path, json.dumps(payload).encode(), h)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", max_audio_bytes=1000)


@pytest.fixture
def server(bot, kg, settings) -> Iterator[Server]:
    with Server(bot, kg, settings) as srv:
        yield srv


def test_health(server: Server):
    status, _, body = server.request("GET", "/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_ask_returns_answer_facts_and_sources(server: Server):
    status, _, body = server.post_json(
        "/ask", {"question": "What does the Premium Savings account include?"}
    )
    data = json.loads(body)
    assert status == 200
    assert "Platinum debit card" in data["answer"]
    assert data["entities"] == ["Premium Savings account"]
    assert data["sources"] and data["facts"]


@pytest.mark.parametrize(
    ("payload", "status"),
    [({}, 422), ({"question": ""}, 422), ({"question": 5}, 422), ({"question": "x" * 1001}, 413)],
)
def test_ask_validation(server: Server, payload, status):
    assert server.post_json("/ask", payload)[0] == status


def test_ask_rejects_bad_json(server: Server):
    status, _, body = server.request("POST", "/ask", b"{nope", {"Content-Type": "application/json"})
    assert status == 400 and "error" in json.loads(body)
    assert server.request("POST", "/ask", b"[1]")[0] == 400


def test_voice_returns_audio_with_transcript_headers(server: Server):
    question = b"What credit score does the Home Loan need?"
    status, headers, audio = server.request("POST", "/voice", question)
    assert status == 200
    assert headers["content-type"] == "audio/wav" and audio[:4] == b"RIFF"
    assert unquote(headers["x-transcript"]) == question.decode()
    assert "700" in unquote(headers["x-answer"])


def test_voice_rejects_empty_and_oversized_audio(server: Server):
    assert server.request("POST", "/voice", b"")[0] == 400
    assert server.request("POST", "/voice", b"x" * 1001)[0] == 413


def test_ingest_then_ask_and_persist(server: Server, settings: Settings):
    status, _, body = server.post_json(
        "/ingest", {"doc_id": "gold", "text": "The Gold plan includes a smart speaker."}
    )
    assert status == 200 and json.loads(body)["facts"] == 1
    assert settings.graph_path.exists()
    _, _, body = server.post_json("/ask", {"question": "Does the Gold plan include a speaker?"})
    assert "smart speaker" in json.loads(body)["answer"]
    assert server.post_json("/ingest", {"doc_id": "", "text": "x"})[0] == 422


def test_unknown_route_and_wrong_method(server: Server):
    assert server.request("GET", "/nope")[0] == 404
    assert server.request("GET", "/ask")[0] == 405


def test_api_key_is_enforced_except_for_health(bot, kg, settings):
    with Server(bot, kg, replace(settings, api_key="s3cret")) as srv:
        assert srv.request("GET", "/health")[0] == 200
        assert srv.post_json("/ask", {"question": "x"})[0] == 401
        assert srv.post_json("/ask", {"question": "x"}, {"Authorization": "Bearer wrong"})[0] == 401
        ok = srv.post_json(
            "/ask", {"question": "home loan credit score"}, {"Authorization": "Bearer s3cret"}
        )
        assert ok[0] == 200


# ---------------------------------------------------------------------------- CLI
@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, value in {
        "KGV_DATA_DIR": str(tmp_path / "kg"),
        "KGV_EXTRACTOR": "rules",
        "KGV_GENERATOR": "extractive",
        "KGV_ASR": "text",
        "KGV_TTS": "stub",
    }.items():
        monkeypatch.setenv(name, value)
    return tmp_path


def test_cli_ingest_stats_ask_voice(cli_env: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["ingest", str(SAMPLE_DIR), "--reset"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["documents"] == 1

    assert main(["stats"]) == 0
    assert json.loads(capsys.readouterr().out)["entities"] > 0

    assert main(["ask", "What credit score does the Home Loan need?"]) == 0
    assert "700" in capsys.readouterr().out

    assert main(["ask", "--json", "What does the Premium Savings account include?"]) == 0
    assert json.loads(capsys.readouterr().out)["entities"] == ["Premium Savings account"]

    question = cli_env / "q.txt"
    question.write_bytes(b"What is the early closure fee?")
    out = cli_env / "a.wav"
    assert main(["voice", str(question), "-o", str(out)]) == 0
    assert out.read_bytes()[:4] == b"RIFF"


def test_cli_ask_without_ingest_is_a_clean_error(cli_env: Path, capsys):
    assert main(["ask", "hello?"]) == 1
    assert "ingest" in capsys.readouterr().err


def test_cli_ingest_missing_path_is_a_clean_error(cli_env: Path, capsys):
    assert main(["ingest", str(cli_env / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
