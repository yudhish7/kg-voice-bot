import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kgvoicebot.extract import RuleBasedExtractor  # noqa: E402
from kgvoicebot.generator import ExtractiveGenerator, build_prompt  # noqa: E402
from kgvoicebot.graph import KnowledgeGraph  # noqa: E402
from kgvoicebot.ingest import Ingestor  # noqa: E402
from kgvoicebot.retriever import GraphRetriever  # noqa: E402

kg = KnowledgeGraph()
Ingestor(kg, RuleBasedExtractor()).ingest_path(ROOT / "data" / "sample")
retriever = GraphRetriever(kg)

question = "What credit score does the Home Loan need?"
evidence = retriever.retrieve(question)

print("=== The message that would be sent to the AI ===")
print(build_prompt(question, evidence))

print()
print("=== The offline answer (no AI used) ===")
print(ExtractiveGenerator().generate(question, evidence))

class LLMGenerator:
    def generate(self, question, evidence):
        return self._llm.complete(SYSTEM_PROMPT, build_prompt(question, evidence)) or FALLBACK_ANSWER