"""Text embeddings for retrieval.

Anthropic has no embeddings API, so embeddings come from elsewhere. Bills are written in English, but
questions may be in English or Kiswahili, so the real providers are multilingual models:

- "local" (default): intfloat/multilingual-e5-small via sentence-transformers. Runs on a laptop CPU, costs
  nothing per query, and bill text never leaves the machine. Install with `pip install -e '.[rag-local]'`.
- "voyage": Voyage AI's hosted multilingual model, the provider Anthropic's documentation points to for
  embeddings. Needs VOYAGE_API_KEY. Install with `pip install -e '.[rag-voyage]'`.
- "fake": deterministic hashed bag-of-words for tests and CI. No downloads, no keys.

See docs/RAG.md ("Embedding model") for why.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


class Embedder(Protocol):
    name: str  # stored in the index; changing provider or model requires a rebuild
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_WORD = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or shall that the this to under "
    "what when which who will with does do any may not must can".split()
    # Kiswahili function words
    + "na ya wa za la kwa katika ni je nini gani au kama hii hiyo huo vipi kuna".split()
)
# Lets the fake embedder line up a handful of Kiswahili question words with English bill vocabulary, so the
# fixture eval can exercise Kiswahili questions without a real model. It says nothing about how well a real
# multilingual model handles Kiswahili; only the real-corpus eval can tell you that.
_SW_GLOSSARY = {
    "adhabu": "penalty", "faini": "fine", "kodi": "levy", "ushuru": "levy", "ada": "fee",
    "kifungu": "clause", "mpanda": "rider", "waendeshaji": "rider", "mwendeshaji": "rider",
    "abiria": "passenger", "kofia": "helmet", "mafunzo": "training", "usajili": "registration",
    "kusajiliwa": "registration", "maji": "water", "shule": "schools", "hospitali": "hospitals",
    "kaya": "households", "nyumba": "households", "mfuko": "fund", "kuchelewa": "late", "asilimia": "per cent",
    "soko": "market", "masoko": "market", "wafanyabiashara": "traders", "msamaha": "exemptions",
    "kulipishwa": "charged", "kifungo": "imprisonment", "jela": "imprisonment", "pikipiki": "motorcycle",
}


def _tokens(text: str) -> list[str]:
    words = []
    for raw in _WORD.findall(text.lower()):
        word = _SW_GLOSSARY.get(raw, raw)
        for w in word.split():
            if w in _STOPWORDS:
                continue
            if len(w) > 4 and w.endswith("s"):
                w = w[:-1]
            words.append(w)
    return words


class HashingEmbedder:
    """Deterministic feature-hashed bag of words. For tests and CI only: no semantics, no cross-lingual skill."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"fake-hashing-{dim}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """Local multilingual model. The e5 family expects "query: " / "passage: " prefixes."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise RuntimeError("Install the local embedder with: pip install -e '.[rag-local]'") from err
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = f"st:{model}"
        self._e5 = "e5" in model.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([("passage: " if self._e5 else "") + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([("query: " if self._e5 else "") + text])[0]


class VoyageEmbedder:
    """Voyage AI hosted embeddings (reads VOYAGE_API_KEY)."""

    def __init__(self, model: str = "voyage-3.5"):
        try:
            import voyageai
        except ImportError as err:
            raise RuntimeError("Install the Voyage embedder with: pip install -e '.[rag-voyage]'") from err
        self._client = voyageai.Client()
        self._model = model
        self.name = f"voyage:{model}"
        self.dim = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = list(texts[i : i + 64])
            out.extend(self._client.embed(batch, model=self._model, input_type="document").embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], model=self._model, input_type="query").embeddings[0]


PROVIDERS = ("local", "voyage", "fake")


def get_embedder(provider: str = "local", model: str | None = None) -> Embedder:
    if provider == "fake":
        return HashingEmbedder()
    if provider == "local":
        return SentenceTransformerEmbedder(model) if model else SentenceTransformerEmbedder()
    if provider == "voyage":
        return VoyageEmbedder(model) if model else VoyageEmbedder()
    raise ValueError(f"Unknown embedder {provider!r}; choose one of {', '.join(PROVIDERS)}")
