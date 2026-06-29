"""
News reviewer — семантический few-shot отбор документов «как эксперт».

Идея: вместо абстрактного «оцени важность» показываем слабой модели реальные
решения эксперта по СМЫСЛОВО ПОХОЖИМ документам и просим судить по аналогии.

Источник примеров:
  - колонка-эталон «Статус отбора» (выбран/не выбран), когда она есть в выгрузке;
  - либо примеры, размеченные пользователем вручную.

Поток для нового документа:
  эмбеддинг содержания → top-k похожих ВЫБРАННЫХ + top-k похожих НЕВЫБРАННЫХ
  → few-shot промпт → GigaChat → {score 0-100, reasoning}.
"""

import json
import re

from .embeddings import SemanticIndex, embed_texts
from .features import resolve_columns, get_content, get
from .evaluation import is_positive_label


def build_index_from_labeled(rows, colmap=None, gc=None):
    """Построить SemanticIndex из строк, где есть колонка-метка отбора."""
    if colmap is None and rows:
        colmap = resolve_columns(rows[0].keys())
    if "label" not in (colmap or {}):
        raise ValueError("В выгрузке нет колонки-эталона отбора — нельзя построить индекс примеров")

    texts, payloads = [], []
    for idx, row in enumerate(rows):
        content = get_content(row, colmap)
        if not content.strip():
            continue
        label = get(row, colmap, "label", "")
        texts.append(content)
        payloads.append({
            "row_index": idx,
            "is_positive": is_positive_label(label),
            "preview": content[:400],
        })
    index = SemanticIndex()
    index.add_texts(texts, payloads, gc=gc)
    return index


def _format_examples(examples, header):
    if not examples:
        return f"{header}\n(нет похожих примеров)"
    blocks = []
    for i, ex in enumerate(examples, 1):
        preview = ex["payload"]["preview"]
        blocks.append(f"{i}. (сходство {ex['score']:.2f}) \"{preview}\"")
    return f"{header}\n" + "\n".join(blocks)


REVIEW_SYSTEM = """Ты воспроизводишь решения старшего аудитора Сбера об отборе документов.
У аудитора есть неявные критерии. Тебе показаны его реальные решения по СМЫСЛОВО ПОХОЖИМ
документам. Оцени новый документ ПО АНАЛОГИИ с этими решениями, а не по общей «важности».
Отвечай СТРОГО валидным JSON без markdown: {"score": 0-100, "reasoning": "кратко почему"}."""


def build_review_user(content, positive_examples, negative_examples, threshold):
    return f"""{_format_examples(positive_examples, "=== ПОХОЖИЕ документы, которые аудитор ВЫБРАЛ ===")}

{_format_examples(negative_examples, "=== ПОХОЖИЕ документы, которые аудитор НЕ выбрал ===")}

=== НОВЫЙ ДОКУМЕНТ ДЛЯ ОЦЕНКИ ===
\"{content[:8000]}\"

Оцени, выбрал бы аудитор этот документ. Шкала 0-100, порог отбора ~{threshold}.
Чем больше документ похож на ВЫБРАННЫЕ и не похож на НЕвыбранные — тем выше score.
Верни JSON: {{"score": 0, "reasoning": ""}}"""


def _parse_review(raw):
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return float(data.get("score", 0)), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        match = re.search(r"\d+(?:\.\d+)?", raw)
        if match:
            return float(match.group(0)), f"(JSON parse error) {raw[:200]}"
        return 0.0, f"(не удалось извлечь оценку) {raw[:200]}"


def review_document(content, index, gc=None, k=3, threshold=70):
    """Оценить один документ через few-shot по похожим размеченным примерам."""
    if gc is None:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()

    query_vec = embed_texts([content], gc=gc)[0]
    positives = index.search(query_vec=query_vec, k=k, where=lambda p: p["is_positive"])
    negatives = index.search(query_vec=query_vec, k=k, where=lambda p: not p["is_positive"])

    resp = gc.completions(
        [{"role": "system", "content": REVIEW_SYSTEM},
         {"role": "user", "content": build_review_user(content, positives, negatives, threshold)}],
        temperature=0,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    score, reasoning = _parse_review(raw)
    return {
        "score": score,
        "reasoning": reasoning,
        "similar_positive": [e["payload"]["row_index"] for e in positives],
        "similar_negative": [e["payload"]["row_index"] for e in negatives],
    }
