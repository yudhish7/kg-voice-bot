import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.audio.asr import TextPassthroughASR  # noqa: E402
from kgvoicebot.audio.tts import ToneTTS  # noqa: E402
from kgvoicebot.bot import VoiceBot  # noqa: E402
from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.generator import ExtractiveGenerator  # noqa: E402
from kgvoicebot.graph import KnowledgeGraph  # noqa: E402
from kgvoicebot.ingest import Ingestor  # noqa: E402
from kgvoicebot.retriever import GraphRetriever  # noqa: E402

# Build the knowledge base
kg = KnowledgeGraph()
Ingestor(kg, RuleBasedExtractor()).ingest_path(ROOT / "data" / "sample")

# Assemble the bot from four parts
bot = VoiceBot(
    retriever=GraphRetriever(kg),
    generator=ExtractiveGenerator(),
    asr=TextPassthroughASR(),  # pretend "audio" is just text
    tts=ToneTTS(),  # makes a simple beep instead of a voice
)

print("--- Text questions ---")
for question in [
    "What credit score does the Home Loan need?",
    "Who won the cricket world cup?",
    "   ",
]:
    answer = bot.ask_text(question)
    print("Q:", repr(question))
    print("A:", answer.text)
    print()

print("--- A voice question ---")
reply = bot.ask_voice(b"What is the early closure fee?")
print("Heard:  ", reply.transcript)
print("Answer: ", reply.answer.text)
print("Audio:  ", len(reply.audio), "bytes of", reply.content_type)

(ROOT / "answer.wav").write_bytes(reply.audio)
print("Saved the audio to answer.wav")
