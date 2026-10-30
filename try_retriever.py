'import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.graph import KnowledgeGraph  # noqa: E402
from kgvoicebot.ingest import Ingestor  # noqa: E402
from kgvoicebot.retriever import GraphRetriever  # noqa: E402

kg = KnowledgeGraph()
Ingestor(kg, RuleBasedExtractor()).ingest_path(ROOT / "data" / "sample")
retriever = GraphRetriever(kg, hops=2, top_k_chunks=4)

question = "What is the ATM limit on the card that comes with Premium Savings?"
evidence = retriever.retrieve(question)

print("Question:", question)
print()
print("1) Entities found:", evidence.entities)

print()
print("2) Facts selected (nearest first):")
for t in evidence.triples:
    print("  ", t.subject, "|", t.relation, "|", t.object)

print()
print("3) Chunks selected (score, id):")
for chunk, score in evidence.chunks:
    print("  ", round(score, 2), chunk.id)'



python try_retriever.py