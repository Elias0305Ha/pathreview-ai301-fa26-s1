"""Integration test: retrieval -> generation -> parsing, offline."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.chunking.base import Chunk
from ingestion.embeddings.provider import MockEmbeddingProvider
from rag.generator.review_generator import ReviewConfig, ReviewGenerator
from rag.retriever.hybrid import HybridRetriever
from rag.retriever.keyword_search import KeywordSearcher
from rag.retriever.vector_store import VectorStore

PROFILE_PATH = Path(__file__).parent.parent / "fixtures" / "sample_profiles" / "basic_profile.json"

# Must match the order in ReviewGenerator.generate_full_review
SECTION_NAMES = [
    "skills_feedback",
    "projects_feedback",
    "presentation_feedback",
    "gaps_feedback",
    "first_impression",
]


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.metadata['source_id']}_chunk_{chunk.metadata['chunk_index']}"


def _as_vector_record(chunk: Chunk) -> SimpleNamespace:
    """Adapt a Chunk to the attributes VectorStore.add_chunks reads.

    Chunk only carries text and metadata, while add_chunks expects
    id, source_id, chunk_index and section as attributes.
    """
    return SimpleNamespace(
        id=_chunk_id(chunk),
        text=chunk.text,
        source_id=chunk.metadata["source_id"],
        chunk_index=chunk.metadata["chunk_index"],
        section=chunk.metadata.get("section", ""),
    )


def _fake_response(section_name: str) -> str:
    payload = {
        section_name: {
            "summary": f"Fixed feedback for {section_name}.",
            "suggestions": [f"{section_name} suggestion 1", f"{section_name} suggestion 2"],
        }
    }
    return f"```json\n{json.dumps(payload)}\n```"


class FakeChatClient:
    """Stands in for openai.OpenAI; returns one fixed JSON response per section."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        section_name = SECTION_NAMES[len(self.calls)]
        self.calls.append(kwargs)
        message = SimpleNamespace(content=_fake_response(section_name))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.integration
def test_rag_pipeline_retrieval_generation_parsing(
    tmp_path, sample_resume_text, sample_readme_text
):
    """A query flows through hybrid retrieval, generation and output parsing."""
    # Ingest: build chunks and embed them
    readme_intro, readme_stack = sample_readme_text.split("## Tech Stack")
    chunks = [
        Chunk(
            text=sample_resume_text,
            metadata={"source_id": "resume_jane_doe", "chunk_index": 0, "section": "resume"},
        ),
        Chunk(
            text=readme_intro,
            metadata={"source_id": "repo_weather_app", "chunk_index": 0, "section": "overview"},
        ),
        Chunk(
            text="## Tech Stack" + readme_stack,
            metadata={"source_id": "repo_weather_app", "chunk_index": 1, "section": "tech_stack"},
        ),
    ]
    embedder = MockEmbeddingProvider()
    embeddings = embedder.embed([c.text for c in chunks])

    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    vector_store.add_chunks(
        [(_as_vector_record(c), e) for c, e in zip(chunks, embeddings)], "profile_test"
    )

    keyword_searcher = KeywordSearcher()
    keyword_searcher.index(
        [{"id": _chunk_id(c), "text": c.text, "metadata": c.metadata} for c in chunks]
    )

    # Retrieve
    retriever = HybridRetriever(vector_store, keyword_searcher)
    query = "Python FastAPI React experience"
    retrieved_chunks = retriever.retrieve(
        query,
        profile_id="test",
        query_embedding=embedder.embed([query])[0],
        min_score=0.0,
    )

    assert {r["id"] for r in retrieved_chunks} == {_chunk_id(c) for c in chunks}
    retrieved_sources = {r["metadata"]["source_id"] for r in retrieved_chunks}

    # Generate and parse
    profile_data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    generator = ReviewGenerator(
        ReviewConfig(api_key="test", base_url="http://localhost:9/v1", model="test")
    )
    fake_client = FakeChatClient()
    generator.client = fake_client

    sections = generator.generate_full_review(profile_data, retrieved_chunks)

    assert [s.section_name for s in sections] == SECTION_NAMES
    for section in sections:
        assert not section.content.startswith("Error generating")
        # 0.9 and the suggestions only come from the JSON parsing path
        assert section.confidence == 0.9
        assert section.suggestions == [
            f"{section.section_name} suggestion 1",
            f"{section.section_name} suggestion 2",
        ]
        sources_line = section.content.splitlines()[-1]
        assert sources_line.startswith("Sources: ")
        cited = set(sources_line.removeprefix("Sources: ").split(", "))
        assert cited and cited <= retrieved_sources

    assert len(fake_client.calls) == 5
    for call in fake_client.calls:
        prompt = call["messages"][-1]["content"]
        for chunk in retrieved_chunks:
            assert chunk["text"] in prompt
