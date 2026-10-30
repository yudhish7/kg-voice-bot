import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.graph import KnowledgeGraph  # noqa: E402
from kgvoicebot.ingest import Ingestor  # noqa: E402

# 1. Build the whole graph from the bank document
kg = KnowledgeGraph()
Ingestor(kg, RuleBasedExtractor()).ingest_path(ROOT / "data" / "sample")
print("Graph size:", kg.stats())

# 2. Which dots does the question mention?
question = "What is the ATM limit on the card that comes with Premium Savings?"
seeds = kg.link_entities(question)
print()
print("Question:", question)
print("Dots found in the question:", seeds)


# 3. Walk outward from those dots (2 steps)
distances = kg.distances(seeds, hops=2)
print()
print("Dots within 2 steps (name -> steps away):")
for node, steps in sorted(distances.items(), key=lambda item: item[1]):
    print("  ", steps, "step(s):", node)