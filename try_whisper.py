import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.audio.asr import FasterWhisperASR  # noqa: E402

asr = FasterWhisperASR("base")
print("Listening... (the first run downloads the model, please wait)")

for folder in sorted((ROOT / "audio").iterdir()):
    wav = folder / "question.wav"
    txt = folder / "question.txt"
    if not wav.exists():
        continue
    heard = asr.transcribe(wav.read_bytes())
    expected = txt.read_text(encoding="utf-8").strip() if txt.exists() else "(no txt)"
    print()
    print(folder.name)
    print("  expected:", expected)
    print("  heard   :", heard)