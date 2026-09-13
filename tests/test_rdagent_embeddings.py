from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from quant_platform.rdagent_embeddings import (
    _text_chunks,
    embed_complete_documents,
    enable_embedding_document_chunking,
)

pytestmark = pytest.mark.no_database


def test_complete_multilingual_source_is_preserved_and_pooled_by_length():
    document = "def predict(x):\n    return x * 2\n研究经验😀" * 500
    seen = []

    def embed(chunks):
        assert len(chunks) == 1 and len(chunks[0].encode()) <= 2048
        seen.append(chunks[0])
        return [[1.0, float(len(seen) % 3)]]

    result = embed_complete_documents([document], embed)
    assert "".join(seen) == document
    expected = [0.0, 0.0]
    for i, chunk in enumerate(seen, 1):
        expected[0] += len(chunk.encode())
        expected[1] += (i % 3) * len(chunk.encode())
    norm = math.hypot(*expected)
    assert len(result) == 1
    assert result[0] == pytest.approx([value / norm for value in expected])


def test_small_documents_preserve_vectors_and_input_order():
    seen = []

    def embed(chunks):
        seen.extend(chunks)
        return [[float(len(seen)), 3.5]]

    assert embed_complete_documents(["first", "second"], embed) == [[1.0, 3.5], [2.0, 3.5]]
    assert seen == ["first", "second"]
    assert embed_complete_documents([], embed) == []


@pytest.mark.parametrize("document", ["", "a" * 2048, "a" * 2049, "😀" * 1025])
def test_chunk_boundaries_keep_unicode_and_exact_content(document):
    chunks = _text_chunks(document)
    assert "".join(chunks) == document
    assert all(len(chunk.encode()) <= 2048 for chunk in chunks)


@pytest.mark.parametrize("response", [[], [[]], [[math.nan]], [[1.0], [2.0]]])
def test_invalid_provider_vectors_are_not_admitted(response):
    with pytest.raises(ValueError):
        embed_complete_documents(["text"], lambda chunks: response)


def test_inconsistent_dimensions_and_zero_aggregate_fail():
    responses = iter([[[1.0]], [[1.0, 2.0]]])
    with pytest.raises(ValueError, match="dimensions"):
        embed_complete_documents(["x" * 3000], lambda chunks: next(responses))
    with pytest.raises(ValueError, match="invalid vector"):
        embed_complete_documents(["x" * 3000], lambda chunks: [[0.0]])


def test_adapter_binds_once_to_only_the_configured_bigmodel_embedding(monkeypatch):
    seen = []

    class Backend:
        def _create_embedding_inner_function(self, documents):
            seen.append(documents)
            return [[1.0, 2.0] for _ in documents]

    module = ModuleType("rdagent.oai.backend.litellm")
    module.LiteLLMAPIBackend = Backend
    module.LITELLM_SETTINGS = SimpleNamespace(embedding_model="hosted_vllm/embedding-3")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    enable_embedding_document_chunking()
    wrapper = Backend._create_embedding_inner_function
    enable_embedding_document_chunking()
    assert Backend._create_embedding_inner_function is wrapper
    assert len(Backend()._create_embedding_inner_function(input_content_list=["x" * 9938])) == 1
    assert len(seen) == 5
    assert "".join(item[0] for item in seen) == "x" * 9938
    seen.clear()
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://another.invalid/v4")
    assert Backend()._create_embedding_inner_function(["x" * 9938]) == [[1.0, 2.0]]
    assert len(seen) == 1
    seen.clear()
    module.LITELLM_SETTINGS.embedding_model = "openai/text-embedding-3-small"
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    Backend()._create_embedding_inner_function(["x" * 9938])
    assert len(seen) == 1
