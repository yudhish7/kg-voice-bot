from __future__ import annotations

import pytest

from kgvoicebot.chunker import chunk_text
from kgvoicebot.text import content_tokens, normalize, split_sentences


def test_normalize_collapses_case_space_and_edge_punctuation():
    assert normalize("  The  Fiber 300.\n") == "the fiber 300"


def test_content_tokens_drops_stopwords_and_stems_plurals():
    assert content_tokens("What does the plan include?") == ["plan", "include"]
    assert content_tokens("includes routers") == ["include", "router"]


def test_split_sentences_treats_headings_and_bullets_as_boundaries():
    text = "# Title\n\nFirst sentence. Second one!\n- a bullet\nwrapped line\ncontinues here."
    assert split_sentences(text) == [
        "Title.",
        "First sentence.",
        "Second one!",
        "a bullet.",
        "wrapped line continues here.",
    ]


def test_split_sentences_empty():
    assert split_sentences("  \n ") == []


def test_chunks_end_on_sentence_boundaries_and_respect_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = chunk_text(text, "d", max_chars=100, overlap_sentences=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert [c.id for c in chunks] == [f"d:{i}" for i in range(len(chunks))]
    # Nothing lost, nothing duplicated without overlap.
    assert " ".join(c.text for c in chunks) == text


def test_chunk_overlap_repeats_trailing_sentence():
    text = "Alpha is one. Beta is two. Gamma is three. Delta is four."
    chunks = chunk_text(text, "d", max_chars=30, overlap_sentences=1)
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_oversized_sentence_is_kept_whole():
    chunks = chunk_text("A" * 300 + ".", "d", max_chars=50)
    assert len(chunks) == 1 and len(chunks[0].text) == 301


def test_chunk_text_validates_arguments():
    with pytest.raises(ValueError):
        chunk_text("x.", "d", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x.", "d", overlap_sentences=-1)


def test_chunk_text_empty_input():
    assert chunk_text("", "d") == []
