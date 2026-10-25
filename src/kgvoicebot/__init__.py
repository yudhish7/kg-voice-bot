"""kgvoicebot - a voice service bot powered by Knowledge-Graph RAG."""

from .bot import VoiceBot
from .graph import KnowledgeGraph
from .ingest import Ingestor
from .models import Answer, Chunk, Evidence, Triple, VoiceReply
from .retriever import GraphRetriever

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Evidence",
    "GraphRetriever",
    "Ingestor",
    "KnowledgeGraph",
    "Triple",
    "VoiceBot",
    "VoiceReply",
    "__version__",
]
