from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import torch

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class BowConfig:
    max_features: int = 20000
    min_df: int = 1
    lowercase: bool = True
    normalize: bool = True


class BowEmbedder:
    def __init__(self, config: BowConfig, vocab: Optional[dict[str, int]] = None) -> None:
        self.config = config
        self.vocab = vocab or {}

    def fit(self, texts: Iterable[str]) -> dict[str, int]:
        doc_freq = Counter()
        term_freq = Counter()
        for text in texts:
            tokens = _tokenize(text, lowercase=self.config.lowercase)
            term_freq.update(tokens)
            doc_freq.update(set(tokens))

        tokens = [
            token
            for token, freq in term_freq.most_common()
            if doc_freq[token] >= self.config.min_df
        ]
        if self.config.max_features is not None:
            tokens = tokens[: self.config.max_features]

        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        return self.vocab

    def transform(self, texts: list[str]) -> torch.Tensor:
        if not self.vocab:
            raise RuntimeError("BoW vocabulary is empty. Call fit() first or provide a vocab.")

        vocab_size = len(self.vocab)
        vectors = np.zeros((len(texts), vocab_size), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = _tokenize(text, lowercase=self.config.lowercase)
            for token in tokens:
                idx = self.vocab.get(token)
                if idx is not None:
                    vectors[row, idx] += 1.0

        if self.config.normalize:
            vectors = _l2_normalize_rows(vectors)

        return torch.from_numpy(vectors)

    def fit_transform(self, texts: list[str]) -> torch.Tensor:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str) -> None:
        payload = {
            "config": {
                "max_features": self.config.max_features,
                "min_df": self.config.min_df,
                "lowercase": self.config.lowercase,
                "normalize": self.config.normalize,
            },
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str) -> "BowEmbedder":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = BowConfig(**payload["config"])
        return cls(config, vocab=payload["vocab"])


class HFEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 32) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1.0)
                pooled = summed / counts
                embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)


@dataclass
class TextEmbeddingResult:
    embeddings: torch.Tensor
    embedder: BowEmbedder | HFEmbedder
    model_name: str


class TextEmbeddingIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        node_ids: list[str],
        texts: list[str],
        use_faiss: bool,
    ) -> None:
        self.node_ids = node_ids
        self.texts = texts
        self.embeddings = _l2_normalize_rows(embeddings.astype(np.float32))
        self.use_faiss = use_faiss
        self.index = None

        if use_faiss:
            try:
                import faiss
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "FAISS is not installed. Install faiss-cpu or disable --use-faiss."
                ) from exc
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def query(self, vectors: np.ndarray, k: int = 1) -> list[list[tuple[str, float, str]]]:
        vectors = _l2_normalize_rows(vectors.astype(np.float32))
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(vectors, k)
        else:
            scores = vectors @ self.embeddings.T
            indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
            scores = np.take_along_axis(scores, indices, axis=1)

        results: list[list[tuple[str, float, str]]] = []
        for row_scores, row_indices in zip(scores, indices):
            row: list[tuple[str, float, str]] = []
            for score, idx in zip(row_scores, row_indices):
                row.append((self.node_ids[int(idx)], float(score), self.texts[int(idx)]))
            results.append(row)
        return results


def embed_texts(
    texts: list[str],
    model_name: str,
    vocab_path: Optional[str] = None,
    bow_config: Optional[BowConfig] = None,
    device: Optional[str] = None,
) -> TextEmbeddingResult:
    if model_name.lower() == "bow":
        config = bow_config or BowConfig()
        if vocab_path is not None:
            try:
                embedder = BowEmbedder.load(vocab_path)
            except FileNotFoundError:
                embedder = BowEmbedder(config)
                embedder.fit(texts)
                embedder.save(vocab_path)
        else:
            embedder = BowEmbedder(config)
            embedder.fit(texts)
        embeddings = embedder.transform(texts)
        return TextEmbeddingResult(embeddings=embeddings, embedder=embedder, model_name="bow")

    embedder = HFEmbedder(model_name=model_name, device=device)
    embeddings = embedder.embed(texts)
    return TextEmbeddingResult(embeddings=embeddings, embedder=embedder, model_name=model_name)


def update_embeddings_for_nodes(
    embeddings: torch.Tensor,
    indices: list[int],
    texts: list[str],
    embedder: BowEmbedder | HFEmbedder,
) -> torch.Tensor:
    if isinstance(embedder, BowEmbedder):
        new_vecs = embedder.transform(texts)
    else:
        new_vecs = embedder.embed(texts)
    embeddings = embeddings.clone()
    embeddings[torch.tensor(indices, dtype=torch.long)] = new_vecs
    return embeddings


def build_text_index(
    embeddings: torch.Tensor,
    node_ids: list[str],
    texts: list[str],
    use_faiss: bool,
) -> TextEmbeddingIndex:
    return TextEmbeddingIndex(embeddings.cpu().numpy(), node_ids, texts, use_faiss)


def _tokenize(text: str, lowercase: bool) -> list[str]:
    if not text:
        return []
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms
