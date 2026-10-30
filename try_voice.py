import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.audio.asr import FasterWhisperASR  # noqa: E402
from kgvoicebot.audio.tts import EspeakTTS  # noqa: E402
from kgvoicebot.bot import VoiceBot  # noqa: E402
from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.generator import ExtractiveGenerator  # noqa: E402
from kgvoicebot.graph import KnowledgeGraph  # noqa: E402
from kgvoicebot.ingest import Ingestor  # noqa: E402
from kgvoicebot.retriever import GraphRetriever  # noqa: E402

# 1. Knowledge base (the brain)
kg = KnowledgeGraph()
Ingestor(kg, RuleBasedExtractor()).ingest_path(ROOT / "data" / "sample")

# 2. The bot with REAL ears (Whisper) and a REAL mouth (espeak-ng)
bot = VoiceBot(
    retriever=GraphRetriever(kg),
    generator=ExtractiveGenerator(),
    asr=FasterWhisperASR("base"),
    tts=EspeakTTS(),
)

out_dir = ROOT / "replies"
out_dir.mkdir(exist_ok=True)

# 3. Ask every recording
for folder in sorted((ROOT / "audio").iterdir()):
    wav = folder / "question.wav"
    if not wav.exists():
        continue
    reply = bot.ask_voice(wav.read_bytes())
    (out_dir / f"{folder.name}.wav").write_bytes(reply.audio)
    print()
    print(folder.name)
    print("  heard :", reply.transcript)
    print("  answer:", reply.answer.text)
    print("  saved : replies/" + folder.name + ".wav", f"({len(reply.audio)} bytes)")