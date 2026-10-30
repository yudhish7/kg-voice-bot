import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.models import Chunk  # noqa: E402

text = "The Premium Savings account includes a Platinum debit card, free cheque books and priority banking."
chunk = Chunk(id="demo:0", doc_id="demo", text=text)

triples = RuleBasedExtractor().extract(chunk)

print("Sentence:", text)
print()
print("Facts found:", len(triples))
for t in triples:
    print(" ", t.subject, "|", t.relation, "|", t.object)
