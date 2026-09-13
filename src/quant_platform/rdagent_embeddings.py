"""Adapt long CoSTEER knowledge documents to the configured embedding API."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from urllib.parse import urlsplit

# BigModel documents 3072 tokens per embedding-3 input. A byte budget avoids
# relying on another provider's tokenizer, with room for tokenizer framing.
# https://docs.bigmodel.cn/cn/guide/models/embedding/embedding-3
_CHUNK_UTF8_BYTES = 2048


def _text_chunks(text: str) -> list[str]:
    chunks = []
    start = 0
    size = 0
    for index, char in enumerate(text):
        width = len(char.encode("utf-8"))
        if size + width > _CHUNK_UTF8_BYTES:
            chunks.append(text[start:index])
            start = index
            size = 0
        size += width
    chunks.append(text[start:])
    return chunks


def embed_complete_documents(
    documents: list[str],
    embed: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    """Return one vector per complete document, without truncating its content.

    Send one chunk per request so both per-input and aggregate request sizes
    stay small. Pool by UTF-8 content length and normalize for cosine retrieval;
    the original graph node and source text remain unchanged. Short documents
    retain the provider's exact vector, including its original magnitude.
    """
    result = []
    for document in documents:
        chunks = _text_chunks(document)
        weighted: list[float] | None = None
        for chunk in chunks:
            response = embed([chunk])
            if len(response) != 1 or not response[0]:
                raise ValueError("embedding provider returned an invalid document count")
            vector = response[0]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embedding provider returned non-finite values")
            if len(chunks) == 1:
                result.append(vector)
                break
            if weighted is None:
                weighted = [0.0] * len(vector)
            if len(vector) != len(weighted):
                raise ValueError("embedding dimensions changed between document chunks")
            weight = len(chunk.encode("utf-8"))
            for index, value in enumerate(vector):
                weighted[index] += value * weight
        if weighted is not None:
            norm = math.sqrt(sum(value * value for value in weighted))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("embedding document aggregation produced an invalid vector")
            result.append([value / norm for value in weighted])
    return result


def enable_embedding_document_chunking() -> None:
    """Keep upstream credentials/cache/graph handling and adapt only BigModel input."""
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMAPIBackend

    if getattr(LiteLLMAPIBackend, "_quantlab_document_chunking", False):
        return
    original = LiteLLMAPIBackend._create_embedding_inner_function

    def embed_documents(self, input_content_list: list[str]) -> list[list[float]]:
        model = LITELLM_SETTINGS.embedding_model
        base = os.environ.get("HOSTED_VLLM_API_BASE", "")
        if (model != "hosted_vllm/embedding-3"
                or urlsplit(base).hostname != "open.bigmodel.cn"):
            return original(self, input_content_list)
        return embed_complete_documents(
            input_content_list, lambda chunks: original(self, chunks)
        )

    LiteLLMAPIBackend._create_embedding_inner_function = embed_documents
    LiteLLMAPIBackend._quantlab_document_chunking = True
