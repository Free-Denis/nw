"""
Audit-hypothesis pipeline — генерация проверяемых аудиторских гипотез.

Подход: LLM + семантический поиск (без метаданных и regex-эвристик).

Конвейер (узкие шаги под слабый GigaChat):
  Шаг 0. Пользователь задаёт ТЕМУ риска (напр. «мошенничество», «налоги»).
  Шаг 1. Семантический отбор: эмбеддим тему и документы, берём top-N
          наиболее релевантных теме документов (core.embeddings).
  Шаг 2. Кросс-документные паттерны: кластеризация отобранных документов
          по смысловому сходству (похожие = один контрагент/схема).
  Шаг 3. По каждому документу LLM-цепочка: факты → риск → гипотеза.
  Шаг 4. По каждому многодокументному кластеру — отдельная кросс-документная
          гипотеза (паттерн виден только на группе).
  Шаг 5. Ранжирование по существенности и силе улики.

Структура гипотезы фиксирована: риск / красный флаг / проверяемое
утверждение / процедура проверки / область / существенность.
"""

import json
import re
import threading
import traceback

from .embeddings import embed_texts, top_k_similar, cluster_by_similarity
from .features import resolve_columns, get_content, get


# ── In-memory state ──────────────────────────────────────────────
HYPO_JOBS = {}
HYPO_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════
# LLM helpers
# ══════════════════════════════════════════════════════════════════

def _extract_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text).strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


def _gc_json(gc, system_prompt, user_prompt, temperature=0.1):
    resp = gc.completions(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_error": "invalid_json", "_raw": raw}


# ══════════════════════════════════════════════════════════════════
# Шаг 3. Извлечение фактов из документа (только по содержанию)
# ══════════════════════════════════════════════════════════════════

EXTRACT_FACTS_SYSTEM = """Ты — внутренний аудитор. Извлеки из документа ТОЛЬКО факты, ничего не оценивай и не придумывай.
Если факта в тексте нет — оставь пустым/false. Отвечай СТРОГО валидным JSON без markdown."""


def build_extract_facts_user(content, theme):
    return f"""Аудиторская тема, под которую анализируем: {theme}

Текст документа:
\"{content[:8000]}\"

Верни JSON (только то, что прямо следует из текста):
{{
  "decision": "что постановили/предложили одной фразой",
  "amounts": [{{"value": "сумма как в тексте", "purpose": "на что"}}],
  "counterparties": ["контрагенты, если упомянуты"],
  "responsible_persons": ["ответственные за исполнение/подпись"],
  "funding_source": "источник средств, если указан",
  "deviations": ["отклонения от типового порядка, исключения, авансы, ручные согласования"],
  "relevant_to_theme": true
}}"""


# ══════════════════════════════════════════════════════════════════
# Шаг 3. Генерация гипотезы по одному документу (few-shot)
# ══════════════════════════════════════════════════════════════════

HYPOTHESIS_SYSTEM = """Ты — старший внутренний аудитор Сбера. По фактам документа сформулируй ОДНУ
проверяемую аудиторскую гипотезу строго по структуре. Гипотеза должна быть конкретной и
проверяемой документами, а не общим рассуждением. Если оснований для гипотезы нет —
верни {"no_hypothesis": true}. Отвечай СТРОГО JSON без markdown.

Пример хорошей гипотезы:
{
  "risk": "Спонсорская выплата 35 млн ₽ со 100% авансом",
  "red_flag": "100% авансирование и финансирование из адресного резерва в обход типовой процедуры закупки",
  "testable_claim": "По авансу отсутствуют закрывающие акты и подтверждение факта мероприятия",
  "audit_procedure": "Сверить платёж с актами приёма-передачи и фактом мероприятия; проверить основание выплаты из резерва",
  "scope": "Все документы со 100% авансом за период выгрузки",
  "materiality": "высокая"
}"""


def build_hypothesis_user(facts, theme):
    return f"""Аудиторская тема: {theme}
Факты документа: {json.dumps(facts, ensure_ascii=False)}

Сформулируй ОДНУ гипотезу в JSON:
{{"risk": "", "red_flag": "", "testable_claim": "", "audit_procedure": "", "scope": "", "materiality": "высокая/средняя/низкая"}}
Если оснований нет — верни {{"no_hypothesis": true}}."""


# ══════════════════════════════════════════════════════════════════
# Шаг 4. Кросс-документная гипотеза по кластеру похожих документов
# ══════════════════════════════════════════════════════════════════

CROSS_HYPOTHESIS_SYSTEM = """Ты — старший внутренний аудитор Сбера. Тебе дана ГРУППА похожих документов.
Найди паттерн, который виден только на группе (повторяющийся контрагент, дробление решений,
систематические авансы, обход лимитов) и сформулируй ОДНУ кросс-документную гипотезу строго
по структуре. Если общего паттерна нет — верни {"no_hypothesis": true}. Отвечай СТРОГО JSON."""


def build_cross_hypothesis_user(cluster_facts, theme):
    docs_block = "\n\n".join(
        f"Документ {i + 1}: {json.dumps(f, ensure_ascii=False)}"
        for i, f in enumerate(cluster_facts)
    )
    return f"""Аудиторская тема: {theme}
Группа из {len(cluster_facts)} смыслово похожих документов (факты):
{docs_block}

Сформулируй ОДНУ кросс-документную гипотезу в JSON:
{{"risk": "", "red_flag": "", "testable_claim": "", "audit_procedure": "", "scope": "", "materiality": "высокая/средняя/низкая"}}
Если общего паттерна нет — верни {{"no_hypothesis": true}}."""


# ══════════════════════════════════════════════════════════════════
# Шаг 5. Ранжирование
# ══════════════════════════════════════════════════════════════════

_MATERIALITY_WEIGHT = {"высокая": 3, "средняя": 2, "низкая": 1}


def _materiality(value):
    return _MATERIALITY_WEIGHT.get(re.sub(r"\s+", " ", str(value or "")).strip().lower(), 1)


def rank_hypotheses(hypotheses):
    """Сортировка: кросс-документные выше, затем существенность, затем релевантность."""
    def score(h):
        return (
            1 if h.get("kind") == "cross_document" else 0,
            _materiality(h.get("materiality")),
            h.get("relevance", 0.0),
        )
    return sorted(hypotheses, key=score, reverse=True)


# ══════════════════════════════════════════════════════════════════
# Фоновый воркер
# ══════════════════════════════════════════════════════════════════

def _new_hypo_job(theme, top_n):
    return {
        "status": "running",
        "theme": theme,
        "top_n": top_n,
        "progress": {"current": 0, "total": 0, "stage": "Запуск..."},
        "clusters": [],
        "hypotheses": [],
        "error": None,
        "cancelled": False,
    }


def _update_progress(job_id, current, total, stage):
    with HYPO_LOCK:
        job = HYPO_JOBS.get(job_id)
        if job:
            job["progress"] = {"current": current, "total": total, "stage": stage}


def _cancelled(job_id):
    with HYPO_LOCK:
        return HYPO_JOBS.get(job_id, {}).get("cancelled", False)


def run_hypothesis_job(job_id, rows, colmap=None, theme="", top_n=15, cluster_threshold=0.82):
    """
    rows — список dict (строки выгрузки). theme — аудиторская тема-запрос.
    Отбираем top_n релевантных теме документов, кластеризуем, генерируем гипотезы.
    """
    try:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()
        if colmap is None and rows:
            colmap = resolve_columns(rows[0].keys())

        contents = [get_content(row, colmap) for row in rows]
        # Краткая карточка каждого документа: на каких документах построена гипотеза
        doc_meta = [
            {
                "row": idx + 1,
                "subject": str(get(row, colmap, "subject", "")) or get_content(row, colmap)[:120],
                "link": str(get(row, colmap, "link", "")),
            }
            for idx, row in enumerate(rows)
        ]

        # Шаг 1. Семантический отбор по теме
        _update_progress(job_id, 0, len(rows), "Эмбеддинг документов...")
        doc_vectors = embed_texts(contents, gc=gc)
        theme_vec = embed_texts([theme], gc=gc)[0]
        ranked = top_k_similar(theme_vec, doc_vectors, k=top_n)
        selected = [(idx, contents[idx], doc_vectors[idx], score) for idx, score in ranked]

        if _cancelled(job_id):
            return

        # Шаг 2. Кросс-документные кластеры среди отобранных
        sel_vectors = [v for _idx, _c, v, _s in selected]
        clusters_local = cluster_by_similarity(sel_vectors, threshold=cluster_threshold)

        # Шаг 3. Факты + гипотеза по каждому отобранному документу
        facts_by_local = {}
        hypotheses = []
        total = len(selected)
        for n, (idx, content, _vec, relevance) in enumerate(selected):
            if _cancelled(job_id):
                return
            _update_progress(job_id, n + 1, total, f"Анализ документа {n + 1} из {total}")

            facts = _gc_json(gc, EXTRACT_FACTS_SYSTEM, build_extract_facts_user(content, theme))
            if "_error" in facts:
                continue
            facts_by_local[n] = facts

            hypo = _gc_json(gc, HYPOTHESIS_SYSTEM, build_hypothesis_user(facts, theme))
            if "_error" in hypo or hypo.get("no_hypothesis"):
                continue
            hypo.update({
                "kind": "single_document",
                "doc_index": idx,
                "relevance": round(relevance, 3),
                "sources": [doc_meta[idx]],
            })
            hypotheses.append(hypo)

        # Шаг 4. Кросс-документная гипотеза по каждому кластеру
        cluster_payloads = []
        for c_i, local_idxs in enumerate(clusters_local):
            if _cancelled(job_id):
                return
            _update_progress(job_id, c_i + 1, len(clusters_local), f"Кластер {c_i + 1} из {len(clusters_local)}")
            cluster_facts = [facts_by_local[li] for li in local_idxs if li in facts_by_local]
            doc_indexes = [selected[li][0] for li in local_idxs]
            cluster_payloads.append({"doc_indexes": doc_indexes, "size": len(local_idxs)})
            if len(cluster_facts) < 2:
                continue
            hypo = _gc_json(gc, CROSS_HYPOTHESIS_SYSTEM, build_cross_hypothesis_user(cluster_facts, theme))
            if "_error" in hypo or hypo.get("no_hypothesis"):
                continue
            hypo.update({
                "kind": "cross_document",
                "doc_indexes": doc_indexes,
                "relevance": 1.0,
                "sources": [doc_meta[i] for i in doc_indexes],
            })
            hypotheses.append(hypo)

        ranked_hypotheses = rank_hypotheses(hypotheses)
        with HYPO_LOCK:
            job = HYPO_JOBS.get(job_id)
            if job:
                job["clusters"] = cluster_payloads
                job["hypotheses"] = ranked_hypotheses
                job["status"] = "done"
                job["progress"]["stage"] = f"Готово: {len(ranked_hypotheses)} гипотез"
    except Exception as exc:
        traceback.print_exc()
        with HYPO_LOCK:
            job = HYPO_JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = str(exc)
