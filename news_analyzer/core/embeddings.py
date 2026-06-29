"""
Semantic retrieval over GigaChat embeddings.

Используется обоими сценариями:
  - Обозреватель: SemanticIndex хранит векторы размеченных примеров;
    для нового документа достаём top-k похожих выбранных/невыбранных → few-shot.
  - Гипотезы: cluster_by_similarity группирует похожие документы (одна тема/
    контрагент) для кросс-документного анализа — без regex/метаданных.

Эмбеддинги считаются через core.gigachat.GigaChatClient.embeddings. Здесь —
только математика поиска/кластеризации и кэш, чтобы не пересчитывать векторы.
"""

import hashlib
import math


# ── Кэш векторов в памяти (ключ — хэш текста) ────────────────────
_EMBED_CACHE = {}


def _key(text, model):
    return hashlib.sha1(f"{model}::{text[:4000]}".encode("utf-8")).hexdigest()


def embed_texts(texts, gc=None, model=None, batch_size=16, use_cache=True):
    """Вернуть list[list[float]] для texts. Кэширует по тексту, батчит запросы."""
    if gc is None:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()
    model = model or getattr(gc, "embedding_model", "default")

    results = [None] * len(texts)
    pending_idx = []
    pending_txt = []
    for i, text in enumerate(texts):
        text = text or ""
        if use_cache:
            cached = _EMBED_CACHE.get(_key(text, model))
            if cached is not None:
                results[i] = cached
                continue
        pending_idx.append(i)
        pending_txt.append(text)

    for start in range(0, len(pending_txt), batch_size):
        chunk_txt = pending_txt[start:start + batch_size]
        chunk_idx = pending_idx[start:start + batch_size]
        vectors = gc.embeddings(chunk_txt, model=model)
        for j, vec in enumerate(vectors):
            results[chunk_idx[j]] = vec
            if use_cache:
                _EMBED_CACHE[_key(chunk_txt[j], model)] = vec
    return results


# ── Косинусное сходство (без numpy, чтобы не плодить зависимости) ──

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(x * x for x in a)) or 1e-12


def cosine(a, b):
    return _dot(a, b) / (_norm(a) * _norm(b))


def top_k_similar(query_vec, vectors, k=4):
    """Вернуть [(index, score)] по убыванию сходства."""
    scored = [(i, cosine(query_vec, v)) for i, v in enumerate(vectors)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


# ── Индекс размеченных примеров для обозревателя ─────────────────

class SemanticIndex:
    """Хранит векторы + произвольные payload'ы (текст, метка, и т.п.)."""

    def __init__(self):
        self.vectors = []
        self.payloads = []

    def add(self, vector, payload):
        self.vectors.append(vector)
        self.payloads.append(payload)

    def add_texts(self, texts, payloads, gc=None, model=None):
        vectors = embed_texts(texts, gc=gc, model=model)
        for vec, payload in zip(vectors, payloads):
            self.add(vec, payload)

    def search(self, query_text=None, query_vec=None, k=4, gc=None, model=None, where=None):
        """Top-k похожих payload'ов. where(payload)->bool — фильтр (напр. по метке)."""
        if query_vec is None:
            query_vec = embed_texts([query_text or ""], gc=gc, model=model)[0]
        candidates = [
            (i, v) for i, v in enumerate(self.vectors)
            if where is None or where(self.payloads[i])
        ]
        scored = [(i, cosine(query_vec, v)) for i, v in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            {"payload": self.payloads[i], "score": score}
            for i, score in scored[:k]
        ]

    def __len__(self):
        return len(self.vectors)


# ── Кластеризация для кросс-документных паттернов ────────────────

def cluster_by_similarity(vectors, threshold=0.82, min_size=2):
    """
    Группировка по косинусному порогу через связные компоненты (single-link).
    Возвращает список кластеров — каждый это список индексов. Только кластеры
    размером >= min_size (один документ — не паттерн).
    """
    n = len(vectors)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [sorted(idxs) for idxs in groups.values() if len(idxs) >= min_size]
    clusters.sort(key=len, reverse=True)
    return clusters
