import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.chunker import chunk_text  # noqa: E402

text = (ROOT / "data" / "sample" / "acme_bank.md").read_text(encoding="utf-8")
chunks = chunk_text(text, "acme_bank.md", max_chars=300, overlap_sentences=1)

print("Number of chunks:", len(chunks))
for chunk in chunks:
    print("-" * 60)
    print(chunk.id, "| length:", len(chunk.text))
    print(chunk.text)