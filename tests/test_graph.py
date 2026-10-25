from __future__ import annotations

from pathlib import Path

import pytest

from kgvoicebot.graph import KnowledgeGraph
from kgvoicebot.models import Chunk, Triple


def _small() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_chunk(Chunk("d:0", "d", "A includes B. B supports C."))
    kg.add_triple(Triple("A", "includes", "B", "d:0"))
    kg.add_triple(Triple("B", "supports", "C", "d:0"))
    return kg


def test_nodes_merge_across_surface_forms():
    kg = KnowledgeGraph()
    assert kg.add_triple(Triple("Fiber 300", "includes", "router", "d:0"))
    assert kg.add_triple(Triple("fiber 300.", "costs", "49 dollars", "d:1"))
    assert kg.stats()["entities"] == 3
    assert kg.mentions("fiber 300") == {"d:0", "d:1"}


def test_duplicate_triple_is_rejected():
    kg = _small()
    assert not kg.add_triple(Triple("A", "includes", "B", "d:0"))
    assert kg.stats()["relations"] == 2


def test_same_fact_from_two_chunks_shares_one_edge():
    kg = _small()
    kg.add_triple(Triple("A", "includes", "B", "d:9"))
    assert kg.graph["a"]["b"]["includes"]["chunks"] == {"d:0", "d:9"}
    assert kg.graph.number_of_edges() == 2


def test_empty_entities_are_ignored():
    assert not KnowledgeGraph().add_triple(Triple("  ", "r", "B", "d:0"))


def test_distances_are_undirected_and_bounded():
    kg = _small()
    assert kg.distances(["a"], hops=2) == {"a": 0, "b": 1, "c": 2}
    assert kg.distances(["c"], hops=1) == {"c": 0, "b": 1}
    assert kg.distances(["missing"], hops=2) == {}


def test_link_entities_exact_longest_match_first():
    kg = KnowledgeGraph()
    kg.add_triple(Triple("Fiber", "is_a", "brand", "d:0"))
    kg.add_triple(Triple("Fiber 300 plan", "includes", "router", "d:0"))
    assert kg.link_entities("Does the Fiber 300 plan include a router?") == [
        "fiber 300 plan",
        "router",
    ]


def test_link_entities_fuzzy_for_abbreviated_names():
    kg = KnowledgeGraph()
    kg.add_triple(Triple("Fiber 1000 plan", "includes", "router", "d:0"))
    assert "fiber 1000 plan" in kg.link_entities("what does fiber 1000 cost")
    # One shared word out of three is not enough.
    assert "fiber 1000 plan" not in kg.link_entities("fiber broadband")


def test_remove_document_drops_chunks_facts_and_orphans():
    kg = _small()
    kg.add_chunk(Chunk("e:0", "e", "X has Y."))
    kg.add_triple(Triple("X", "has", "Y", "e:0"))
    kg.remove_document("d")
    assert set(kg.chunks) == {"e:0"}
    assert kg.stats() == {"chunks": 1, "entities": 2, "relations": 1}
    kg.remove_document("nope")  # no-op


def test_version_changes_on_mutation():
    kg = KnowledgeGraph()
    v0 = kg.version
    kg.add_triple(Triple("A", "r", "B", "d:0"))
    assert kg.version > v0


def test_save_and_load_round_trip(tmp_path: Path):
    kg = _small()
    path = tmp_path / "nested" / "graph.json"
    kg.save(path)
    assert not path.with_suffix(".json.tmp").exists()
    loaded = KnowledgeGraph.load(path)
    assert loaded.stats() == kg.stats()
    assert loaded.triples == kg.triples
    assert loaded.chunks == kg.chunks


def test_load_rejects_unknown_format():
    with pytest.raises(ValueError, match="format"):
        KnowledgeGraph.from_dict({"format": 99, "chunks": [], "triples": []})
