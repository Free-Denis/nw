"""
Audit-hypothesis pipeline — генерация проверяемых аудиторских гипотез.

Подход (bottom-up, без эмбеддингов, без начальной темы):
  Этап 1 (map). По КАЖДОМУ документу glm-5.1 строит «карточку»: суть,
      риск-сигналы, контрагенты, лица, суммы, ссылки на документы, схема,
      риск-теги из единого управляемого списка.
  Этап 2 (код). Поиск совпадений: документы группируются по общим ключам
      (контрагент, базовый документ, риск-тег, схема). Группа = совпадение.
  Этап 3 (reduce, glm-5.1). По каждой группе модель ищет аудиторский паттерн
      и формулирует гипотезу (или отбрасывает). Сильные одиночки — тоже.
  Этап 4. Ранжирование/дедуп.

Структура гипотезы: риск / красный флаг / проверяемое утверждение /
процедура проверки / где искать / существенность + документы-основания.
"""

import json
import re
import threading
import traceback
from collections import defaultdict

from .features import resolve_columns, get_content, get


# ── In-memory state ──────────────────────────────────────────────
HYPO_JOBS = {}
HYPO_LOCK = threading.Lock()


# Единый список риск-тегов — основа смысловой группировки (вместо эмбеддингов).
RISK_TAGS = [
    "закупки", "налоги", "связанные стороны", "авансирование",
    "обход лимитов или дробление", "спонсорство и благотворительность",
    "кадровые выплаты и льготы", "недвижимость и активы",
    "конфликт интересов", "гарантии и поручительства",
    "льготные условия контрагенту", "изменение ранее принятых решений",
]


# ══════════════════════════════════════════════════════════════════
# LLM helpers
# ══════════════════════════════════════════════════════════════════

def _extract_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text).strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


def _gc_json(gc, system_prompt, user_prompt, model=None, temperature=0.1):
    resp = gc.completions(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=temperature,
        model=model,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_error": "invalid_json", "_raw": raw}


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _norm_list(value):
    out = []
    if isinstance(value, list):
        for v in value:
            n = _norm(v)
            if n and len(n) >= 3:
                out.append(n)
    return out


# ══════════════════════════════════════════════════════════════════
# Этап 1. Карточка документа
# ══════════════════════════════════════════════════════════════════

CARD_SYSTEM = """Ты — внутренний аудитор. Разбери документ и извлеки ТОЛЬКО факты, ничего не выдумывай.
Если факта нет — оставь пустым/[]. Отвечай СТРОГО валидным JSON без markdown."""


def build_card_user(content):
    tags = ", ".join(RISK_TAGS)
    return f"""Документ:
\"{content[:8000]}\"

Доступные риск-теги (выбери 0–3 подходящих, ровно как в списке): {tags}

Верни JSON:
{{
  "summary": "суть документа в 1-2 предложениях",
  "risk_signals": ["конкретные красные флаги, если есть"],
  "counterparties": ["контрагенты/организации, если упомянуты"],
  "persons": ["ФИО ответственных лиц"],
  "amounts": [{{"value": "сумма как в тексте", "purpose": "на что"}}],
  "referenced_docs": ["реквизиты упомянутых/изменяемых документов"],
  "scheme": "короткая метка типа операции",
  "risk_tags": ["тег из списка"]
}}"""


# ══════════════════════════════════════════════════════════════════
# Этап 2. Поиск совпадений (код)
# ══════════════════════════════════════════════════════════════════

_LINK_LABELS = {
    "counterparty": "общий контрагент",
    "referenced_doc": "ссылка на один документ",
    "risk_tag": "общий риск-тег",
    "scheme": "одинаковая схема",
}


def _group_cards(cards):
    """cards: [{idx, card}]. Вернуть группы [{doc_indexes, links}] (size>=2), без дублей по составу."""
    buckets = {k: defaultdict(list) for k in _LINK_LABELS}
    for item in cards:
        idx, c = item["idx"], item["card"]
        for cp in _norm_list(c.get("counterparties")):
            buckets["counterparty"][cp].append(idx)
        for rd in _norm_list(c.get("referenced_docs")):
            buckets["referenced_doc"][rd].append(idx)
        for tg in _norm_list(c.get("risk_tags")):
            buckets["risk_tag"][tg].append(idx)
        sch = _norm(c.get("scheme"))
        if sch and len(sch) >= 4:
            buckets["scheme"][sch].append(idx)

    # собираем группы, дедуплицируем по составу документов
    by_docset = {}
    for ktype, d in buckets.items():
        for key, idxs in d.items():
            idxs = tuple(sorted(set(idxs)))
            if len(idxs) < 2:
                continue
            link = f"{_LINK_LABELS[ktype]}: {key}"
            if idxs in by_docset:
                by_docset[idxs]["links"].append(link)
            else:
                by_docset[idxs] = {"doc_indexes": list(idxs), "links": [link]}
    groups = list(by_docset.values())
    groups.sort(key=lambda g: len(g["doc_indexes"]), reverse=True)
    return groups


# ══════════════════════════════════════════════════════════════════
# Этап 3. Гипотезы
# ══════════════════════════════════════════════════════════════════

GROUP_HYPOTHESIS_SYSTEM = """Ты — старший внутренний аудитор Сбера. Тебе дана ГРУППА связанных документов.
Найди аудиторский паттерн, видимый на группе (повторяющийся контрагент, дробление решений,
систематические авансы, обход лимитов, льготные условия) и сформулируй ОДНУ проверяемую
гипотезу строго по структуре. Если общего паттерна нет — верни {"no_hypothesis": true}.
Отвечай СТРОГО JSON без markdown."""

SINGLE_HYPOTHESIS_SYSTEM = """Ты — старший внутренний аудитор Сбера. По карточке одного документа сформулируй
ОДНУ проверяемую аудиторскую гипотезу строго по структуре, если есть реальное основание.
Если оснований нет — верни {"no_hypothesis": true}. Отвечай СТРОГО JSON без markdown.

Пример:
{"risk":"Спонсорская выплата 35 млн ₽ со 100% авансом","red_flag":"100% аванс и оплата из резерва в обход типовой процедуры","testable_claim":"По авансу нет закрывающих актов и подтверждения мероприятия","audit_procedure":"Сверить платёж с актами и фактом мероприятия","scope":"Документы со 100% авансом за период","materiality":"высокая"}"""

_HYPO_FIELDS = '{"risk":"","red_flag":"","testable_claim":"","audit_procedure":"","scope":"","materiality":"высокая/средняя/низкая"}'


def build_group_hypothesis_user(group_cards, links):
    docs_block = "\n\n".join(
        f"Документ {i + 1}: {json.dumps(c, ensure_ascii=False)}"
        for i, c in enumerate(group_cards)
    )
    return f"""Что связывает документы: {"; ".join(links)}
Карточки документов группы:
{docs_block}

Сформулируй ОДНУ кросс-документную гипотезу в JSON:
{_HYPO_FIELDS}
Если общего паттерна нет — верни {{"no_hypothesis": true}}."""


def build_single_hypothesis_user(card):
    return f"""Карточка документа: {json.dumps(card, ensure_ascii=False)}

Сформулируй ОДНУ гипотезу в JSON:
{_HYPO_FIELDS}
Если оснований нет — верни {{"no_hypothesis": true}}."""


# ══════════════════════════════════════════════════════════════════
# Этап 4. Ранжирование
# ══════════════════════════════════════════════════════════════════

_MATERIALITY_WEIGHT = {"высокая": 3, "средняя": 2, "низкая": 1}


def rank_hypotheses(hypotheses):
    def score(h):
        return (
            1 if h.get("kind") == "cross_document" else 0,
            _MATERIALITY_WEIGHT.get(_norm(h.get("materiality")), 1),
            len(h.get("sources", [])),
        )
    return sorted(hypotheses, key=score, reverse=True)


# ══════════════════════════════════════════════════════════════════
# Фоновый воркер
# ══════════════════════════════════════════════════════════════════

def _new_hypo_job():
    return {
        "status": "running",
        "progress": {"current": 0, "total": 0, "stage": "Запуск..."},
        "groups": [],
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


def _row_content(row, colmap, content_column):
    """Текст документа: из выбранного пользователем столбца либо авто (Содержание+Текст)."""
    if content_column and content_column in row:
        val = row.get(content_column)
        text = "" if val is None else str(val).strip()
        if text and text.lower() != "nan":
            return text
    return get_content(row, colmap)


def run_hypothesis_job(job_id, rows, colmap=None, model=None, content_column=None):
    """Разобрать все документы, найти совпадения, сгенерировать гипотезы из групп."""
    try:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()
        if colmap is None and rows:
            colmap = resolve_columns(rows[0].keys())

        # карточка каждого документа для отображения источников
        doc_meta = [
            {
                "row": idx + 1,
                "subject": str(get(row, colmap, "subject", "")) or _row_content(row, colmap, content_column)[:120],
                "link": str(get(row, colmap, "link", "")),
            }
            for idx, row in enumerate(rows)
        ]

        # ── Этап 1: карточки по всем документам ──
        total = len(rows)
        cards = []
        for idx, row in enumerate(rows):
            if _cancelled(job_id):
                return
            _update_progress(job_id, idx + 1, total, f"Разбор документа {idx + 1} из {total}")
            content = _row_content(row, colmap, content_column)
            if not content.strip():
                continue
            card = _gc_json(gc, CARD_SYSTEM, build_card_user(content), model=model)
            if "_error" in card:
                continue
            cards.append({"idx": idx, "card": card})

        # ── Этап 2: поиск совпадений (код) ──
        _update_progress(job_id, total, total, "Поиск совпадений между документами...")
        groups = _group_cards(cards)
        grouped_idx = {i for g in groups for i in g["doc_indexes"]}
        card_by_idx = {item["idx"]: item["card"] for item in cards}

        with HYPO_LOCK:
            if job_id in HYPO_JOBS:
                HYPO_JOBS[job_id]["groups"] = [
                    {"doc_indexes": g["doc_indexes"], "links": g["links"]} for g in groups
                ]

        hypotheses = []

        # ── Этап 3a: гипотезы по группам ──
        for n, g in enumerate(groups, 1):
            if _cancelled(job_id):
                return
            _update_progress(job_id, n, len(groups), f"Гипотеза по группе {n} из {len(groups)}")
            group_cards = [card_by_idx[i] for i in g["doc_indexes"][:12]]
            hypo = _gc_json(gc, GROUP_HYPOTHESIS_SYSTEM,
                            build_group_hypothesis_user(group_cards, g["links"]), model=model)
            if "_error" in hypo or hypo.get("no_hypothesis"):
                continue
            hypo.update({
                "kind": "cross_document",
                "doc_indexes": g["doc_indexes"],
                "links": g["links"],
                "sources": [doc_meta[i] for i in g["doc_indexes"]],
            })
            hypotheses.append(hypo)

        # ── Этап 3b: сильные одиночки (есть риск-сигналы, не попали в группы) ──
        singles = [
            item for item in cards
            if item["idx"] not in grouped_idx and item["card"].get("risk_signals")
        ]
        for n, item in enumerate(singles, 1):
            if _cancelled(job_id):
                return
            _update_progress(job_id, n, len(singles), f"Гипотеза по одиночному документу {n} из {len(singles)}")
            hypo = _gc_json(gc, SINGLE_HYPOTHESIS_SYSTEM,
                            build_single_hypothesis_user(item["card"]), model=model)
            if "_error" in hypo or hypo.get("no_hypothesis"):
                continue
            hypo.update({
                "kind": "single_document",
                "doc_index": item["idx"],
                "sources": [doc_meta[item["idx"]]],
            })
            hypotheses.append(hypo)

        ranked = rank_hypotheses(hypotheses)
        with HYPO_LOCK:
            job = HYPO_JOBS.get(job_id)
            if job:
                job["hypotheses"] = ranked
                job["status"] = "done"
                job["progress"]["stage"] = f"Готово: {len(ranked)} гипотез из {len(cards)} документов"
    except Exception as exc:
        traceback.print_exc()
        with HYPO_LOCK:
            job = HYPO_JOBS.get(job_id)
            if job:
                job["status"] = "error"
                job["error"] = str(exc)
