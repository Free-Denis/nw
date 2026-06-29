"""
Prompt Laboratory — Reverse Prompt Engineering module.

Multi-step wizard that generates optimal prompts from user examples:
  Step 1: Collect examples (positive/negative) + user intent
  Step 2: AI analyzes each document, then consolidates patterns
  Step 3: AI generates clarifying questions (optional)
  Step 4: Generate prompt → test → evaluate (100pt) → iterate
"""

import copy
import json
import re
import threading
import traceback
import uuid

from .api_queue import API_QUEUE, ApiQueueTask


# ── In-memory state ──────────────────────────────────────────────
LAB_JOBS = {}
LAB_LOCK = threading.Lock()


# ── Helper: extract JSON from GigaChat response ─────────────────

def _extract_json(text):
    """Parse JSON from GigaChat response, stripping markdown fences."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    return json.loads(cleaned)


def _gc_call(gc, system_prompt, user_prompt, temperature=0.1):
    """Single GigaChat completion call, returns parsed JSON."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = gc.completions(messages, temperature=temperature)
    raw = resp["choices"][0]["message"]["content"].strip()
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_error": "invalid_json", "_raw": raw}


def _gc_call_text(gc, system_prompt, user_prompt, temperature=0.1):
    """Single GigaChat completion call, returns raw text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = gc.completions(messages, temperature=temperature)
    return resp["choices"][0]["message"]["content"].strip()


def _parse_score(value):
    """Return numeric score from model output, or None when it is unusable."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _label_name(label):
    return "ВАЖНЫЙ" if label == "positive" else "НЕВАЖНЫЙ"


def _format_examples_for_prompt(examples, max_items=12, max_chars=700):
    rows = []
    for idx, example in enumerate(examples[:max_items], 1):
        rows.append(
            f"{idx}. Ожидаемый класс: {_label_name(example.get('label'))}\n"
            f"Текст: \"{example.get('text', '')[:max_chars]}\""
        )
    return "\n\n".join(rows) or "Нет примеров"


def _build_test_diagnostics(test_results, threshold):
    false_negatives = []
    false_positives = []
    correct = []
    invalid = []

    for idx, result in enumerate(test_results, 1):
        score = _parse_score(result.get("score"))
        expected_positive = result.get("label") == "positive"
        actual_positive = score is not None and score >= threshold
        item = {
            "index": idx,
            "expected": _label_name(result.get("label")),
            "score": result.get("score"),
            "text": result.get("text", "")[:900],
            "reasoning": result.get("reasoning", "")[:500],
        }
        if score is None:
            invalid.append(item)
        elif expected_positive and not actual_positive:
            false_negatives.append(item)
        elif not expected_positive and actual_positive:
            false_positives.append(item)
        else:
            correct.append(item)

    total = len(test_results)
    error_count = len(false_negatives) + len(false_positives) + len(invalid)
    accuracy = round(((total - error_count) / total) * 100, 1) if total else 0
    return {
        "threshold": threshold,
        "total": total,
        "accuracy": accuracy,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "invalid_outputs": invalid,
        "correct": correct,
    }


def _diagnostics_text(diagnostics):
    def lines(items):
        if not items:
            return "Нет"
        return "\n\n".join(
            f"- #{item['index']} ожидалось: {item['expected']}, score: {item['score']}\n"
            f"  Текст: \"{item['text']}\"\n"
            f"  Обоснование модели: \"{item['reasoning']}\""
            for item in items
        )

    return f"""Порог важности: {diagnostics['threshold']}
Всего тестов: {diagnostics['total']}
Точность по размеченным примерам: {diagnostics['accuracy']}%

FALSE NEGATIVE: важные документы ошибочно оценены ниже порога:
{lines(diagnostics['false_negatives'])}

FALSE POSITIVE: неважные документы ошибочно оценены выше или равны порогу:
{lines(diagnostics['false_positives'])}

Некорректный JSON/score:
{lines(diagnostics['invalid_outputs'])}
"""


def _calibrate_evaluation(evaluation, diagnostics):
    """Keep model evaluation, but make the total reflect real classification errors."""
    if not isinstance(evaluation, dict):
        evaluation = {}

    def component(name):
        parsed = _parse_score(evaluation.get(name))
        return int(parsed) if parsed is not None else 0

    total = diagnostics.get("total", 0)
    errors = (
        len(diagnostics.get("false_negatives", []))
        + len(diagnostics.get("false_positives", []))
        + len(diagnostics.get("invalid_outputs", []))
    )
    if total:
        separation = max(0, round(25 * (total - errors) / total))
    else:
        separation = 0

    model_separation = component("separation_quality")
    evaluation["separation_quality"] = min(model_separation or separation, separation)
    evaluation["diagnostics"] = diagnostics

    model_total = component("total")
    component_total = (
        evaluation["separation_quality"]
        + component("reasoning_quality")
        + component("criteria_coverage")
        + component("consistency")
    )
    if errors:
        evaluation["total"] = min(model_total, component_total, 89)
    else:
        evaluation["total"] = min(model_total or component_total, component_total or 100)
    return evaluation


def _as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _item_text(item):
    if isinstance(item, dict):
        name = item.get("name") or item.get("case") or item.get("rule") or item.get("criterion") or item.get("description") or ""
        description = item.get("description") or item.get("decision") or item.get("when") or item.get("condition") or ""
        weight = item.get("weight", item.get("penalty", ""))
        weight_text = f" ({weight} баллов)" if weight not in ("", None) else ""
        if name and description and name != description:
            return f"{name}{weight_text}: {description}"
        return f"{name or description}{weight_text}".strip()
    return str(item)


def _section_lines(items, empty="Не выявлено"):
    rows = [_item_text(item) for item in _as_list(items)]
    rows = [row for row in rows if row]
    if not rows:
        return f"- {empty}"
    return "\n".join(f"- {row}" for row in rows)


def _normalize_policy(policy, user_input, consolidation, threshold):
    """Return a stable decision-policy shape even if the model omits fields."""
    if not isinstance(policy, dict):
        policy = {}

    positive = _as_list(policy.get("positive_signals"))
    if not positive:
        positive = [
            {"name": str(item), "description": str(item), "weight": 0}
            for item in _as_list(consolidation.get("draft_criteria"))
        ]

    negative = _as_list(policy.get("negative_signals"))
    if not negative:
        negative = [
            {"name": str(item), "description": str(item), "penalty": 0}
            for item in _as_list(consolidation.get("negative_patterns"))
        ]

    return {
        "task_summary": policy.get("task_summary") or user_input.get("user_comment", ""),
        "important_definition": policy.get("important_definition") or "Документ важен, если он соответствует целевой задаче пользователя и набирает проходной балл по рубрике.",
        "unimportant_definition": policy.get("unimportant_definition") or "Документ неважен, если он не содержит достаточных признаков ценности для целевой аудитории или подпадает под исключения.",
        "threshold": policy.get("threshold") or threshold,
        "positive_signals": positive,
        "negative_signals": negative,
        "hard_rejects": _as_list(policy.get("hard_rejects")),
        "scoring_rubric": _as_list(policy.get("scoring_rubric") or positive),
        "conflict_rules": _as_list(policy.get("conflict_rules")),
        "edge_cases": _as_list(policy.get("edge_cases")),
        "calibration_rules": _as_list(policy.get("calibration_rules")),
        "evidence_notes": _as_list(policy.get("evidence_notes")),
        "changes_made": _as_list(policy.get("changes_made")),
    }


def _compile_policy_to_prompt(policy, user_input, threshold):
    topics = ", ".join(user_input.get("topics", [])) or "не заданы"
    audience = user_input.get("target_audience") or "не задана"
    scale = user_input.get("scale") or "не задан"
    task = policy.get("task_summary") or user_input.get("user_comment", "")

    return f"""Ты — эксперт по оценке важности корпоративных документов.

ЗАДАЧА
Оцени, насколько документ важен для пользователя.
Контекст задачи: {task}
Целевая аудитория: {audience}
Масштаб интереса: {scale}
Темы интереса: {topics}

ОПРЕДЕЛЕНИЕ ВАЖНОГО ДОКУМЕНТА
{policy.get('important_definition')}

ОПРЕДЕЛЕНИЕ НЕВАЖНОГО ДОКУМЕНТА
{policy.get('unimportant_definition')}

ПОРОГ РЕШЕНИЯ
- Используй шкалу 0-100.
- Score >= {threshold}: документ считается важным.
- Score < {threshold}: документ считается неважным.
- Не ставь высокий score только за совпадение темы: документ должен иметь практическую значимость для аудитории.

СИЛЬНЫЕ ПОЗИТИВНЫЕ СИГНАЛЫ
{_section_lines(policy.get('positive_signals'))}

НЕГАТИВНЫЕ СИГНАЛЫ И ПОНИЖАЮЩИЕ ПРИЗНАКИ
{_section_lines(policy.get('negative_signals'))}

HARD REJECT: ЕСЛИ ВЫПОЛНЯЕТСЯ ЛЮБОЕ ПРАВИЛО, SCORE ДОЛЖЕН БЫТЬ НИЖЕ {threshold}
{_section_lines(policy.get('hard_rejects'), 'Нет абсолютных исключений')}

100-БАЛЛЬНАЯ РУБРИКА
Оцени документ по этим критериям. Если критерии конфликтуют, применяй правила конфликтов ниже.
{_section_lines(policy.get('scoring_rubric'))}

ПРАВИЛА КОНФЛИКТОВ
{_section_lines(policy.get('conflict_rules'), 'При конфликте сильного позитивного и негативного сигнала снижай score до пограничной зоны и объясняй причину.')}

СПОРНЫЕ СЛУЧАИ
{_section_lines(policy.get('edge_cases'), 'Нет отдельных спорных случаев')}

КАЛИБРОВКА
{_section_lines(policy.get('calibration_rules'), f'Похожие на позитивные примеры документы должны быть >= {threshold}; похожие на негативные должны быть < {threshold}.')}

ФОРМАТ ОТВЕТА
Ответь строго валидным JSON без markdown и без дополнительного текста:
{{
  "score": 0,
  "reasoning": "Кратко: какие позитивные и негативные признаки найдены, какие правила сработали, почему итоговый score выше или ниже порога."
}}

ТРЕБОВАНИЯ К ОЦЕНКЕ
- score должен быть числом от 0 до 100.
- reasoning должен быть кратким прикладным объяснением, без скрытых рассуждений и без длинной пошаговой цепочки.
- Если документ пустой, поврежденный, является заглушкой или не содержит оценимого содержания, верни score 0."""


# ── Step 2: Analyze each document individually ───────────────────

ANALYZE_SINGLE_SYSTEM = """Ты — аналитик корпоративных документов. Проанализируй документ и определи его ключевые характеристики.
ВНИМАНИЕ: Если текст является сообщением об ошибке, заглушкой о конфиденциальности, обрывком текста без смысла или пустым, обязательно верни "is_broken": true.
Ответь СТРОГО в формате JSON без дополнительного текста."""

def _build_analyze_single_user(text, label, topics):
    label_text = "ВАЖНЫЙ (позитивный пример)" if label == "positive" else "НЕВАЖНЫЙ (негативный пример)"
    topics_text = ", ".join(topics) if topics else "не указаны"
    return f"""Документ (отмечен пользователем как {label_text}):
\"{text[:6000]}\"

Пользователь ищет документы по темам: {topics_text}.

Ответь строго в JSON:
{{
  "is_broken": false,
  "main_topic": "основная тема документа",
  "subtopics": ["подтема1", "подтема2"],
  "key_entities": ["персона1", "подразделение1"],
  "financial_impact": "есть/нет, описание",
  "has_deadlines": true,
  "urgency": "высокая/средняя/низкая",
  "document_type": "поручение/записка/постановление",
  "scale": "отдел/подразделение/компания/отрасль",
  "why_important_or_not": "Почему этот документ важен или неважен с точки зрения заданных тем",
  "distinguishing_features": "Что отличает этот документ"
}}"""


# ── Step 2b: Consolidate all analyses ────────────────────────────

CONSOLIDATE_SYSTEM = """Ты — эксперт по промпт-инжинирингу. На основе анализов отдельных документов найди закономерности, которые отличают ВАЖНЫЕ документы от НЕВАЖНЫХ. Ответь СТРОГО в формате JSON без дополнительного текста."""

def _build_consolidate_user(analyses, user_input):
    positive_docs = []
    negative_docs = []
    for item in analyses:
        entry = f"Текст (начало): \"{item['text'][:300]}...\"\nАнализ: {json.dumps(item['analysis'], ensure_ascii=False)}"
        if item["label"] == "positive":
            positive_docs.append(entry)
        else:
            negative_docs.append(entry)

    pos_text = "\n\n".join(f"Документ {i+1}:\n{d}" for i, d in enumerate(positive_docs)) or "Нет позитивных примеров"
    neg_text = "\n\n".join(f"Документ {i+1}:\n{d}" for i, d in enumerate(negative_docs)) or "Нет негативных примеров"

    return f"""Контекст пользователя: \"{user_input.get('user_comment', '')}\"
Целевая аудитория: \"{user_input.get('target_audience', '')}\"
Масштаб: \"{user_input.get('scale', '')}\"
Темы: {json.dumps(user_input.get('topics', []), ensure_ascii=False)}

=== ВАЖНЫЕ документы (позитивные примеры) ===
{pos_text}

=== НЕВАЖНЫЕ документы (негативные примеры) ===
{neg_text}

Ответь в JSON:
{{
  "positive_patterns": ["Паттерн 1: что общего у ВАЖНЫХ документов", "Паттерн 2: ..."],
  "negative_patterns": ["Паттерн 1: что общего у НЕВАЖНЫХ документов"],
  "key_differentiators": ["Главное отличие 1: ВАЖНЫЕ имеют X, а НЕВАЖНЫЕ — нет"],
  "detected_topics": ["тема1", "тема2"],
  "scale_assessment": "описание масштаба, который важен пользователю",
  "blind_spots": ["Неясно 1: ...", "Неясно 2: ..."],
  "draft_criteria": ["Критерий 1: ...", "Критерий 2: ..."]
}}"""


# ── Step 3: Generate clarifying questions ────────────────────────

QUESTIONS_SYSTEM = """Ты — эксперт по промпт-инжинирингу. На основе анализа примеров у тебя есть черновые критерии оценки. Сформулируй 3-5 уточняющих вопросов, ответы на которые помогут создать точный промпт.

Каждый вопрос должен:
- Закрывать конкретную слепую зону
- Быть закрытым (с вариантами ответа) для удобства
- Один из вопросов ОБЯЗАТЕЛЬНО про масштаб влияния
- Один из вопросов ОБЯЗАТЕЛЬНО про конкретные ситуации/исключения

Ответь СТРОГО в формате JSON без дополнительного текста."""

def _build_questions_user(consolidation, user_input):
    return f"""Контекст пользователя: \"{user_input.get('user_comment', '')}\"
Выявленные слепые зоны: {json.dumps(consolidation.get('blind_spots', []), ensure_ascii=False)}
Текущие критерии: {json.dumps(consolidation.get('draft_criteria', []), ensure_ascii=False)}
Масштаб: \"{user_input.get('scale', '')}\"
Паттерны важных: {json.dumps(consolidation.get('positive_patterns', []), ensure_ascii=False)}
Паттерны неважных: {json.dumps(consolidation.get('negative_patterns', []), ensure_ascii=False)}

Ответь в JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "text": "Текст вопроса",
      "type": "single_choice",
      "options": ["Вариант A", "Вариант B", "Вариант C"],
      "why": "Зачем этот вопрос нужен"
    }}
  ]
}}"""


# ── Step 4a: Build decision policy, then compile prompt ──────────

POLICY_SYSTEM = """Ты — архитектор decision policy для LLM-классификатора.
Твоя задача — не писать финальный промпт. Твоя задача — восстановить явную модель принятия решения по размеченным примерам.

Создай policy, которую потом можно механически скомпилировать в промпт.

Правила:
- Используй только признаки, которые подтверждаются задачей пользователя, примерами или анализом примеров.
- Разделяй позитивные сигналы, негативные сигналы, hard reject, конфликтные правила и спорные случаи.
- Рубрика должна быть 100-балльной. Позитивные критерии должны иметь веса, сумма основных позитивных весов должна быть близка к 100.
- Негативные сигналы должны быть штрафами или правилами ограничения максимального score.
- Hard reject должен опускать документ ниже пользовательского порога.
- В evidence_notes укажи, какие паттерны из примеров поддерживают правила.
- Не создавай финальный prompt_text.

Ответь СТРОГО JSON без дополнительного текста:
{
  "task_summary": "...",
  "important_definition": "...",
  "unimportant_definition": "...",
  "threshold": 70,
  "positive_signals": [{"name": "...", "description": "...", "weight": 25, "evidence": "..."}],
  "negative_signals": [{"name": "...", "description": "...", "penalty": -20, "evidence": "..."}],
  "hard_rejects": ["..."],
  "scoring_rubric": [{"criterion": "...", "weight": 25, "how_to_score": "..."}],
  "conflict_rules": ["..."],
  "edge_cases": [{"case": "...", "decision": "..."}],
  "calibration_rules": ["..."],
  "evidence_notes": ["..."]
}"""

def _build_policy_user(consolidation, user_input, answers=None):
    answers_text = ""
    if answers:
        answers_text = "\n\nУточнения от пользователя:\n"
        for qid, answer in answers.items():
            answers_text += f"- {qid}: {answer}\n"

    return f"""Запрос пользователя: \"{user_input.get('user_comment', '')}\"
Целевая аудитория: \"{user_input.get('target_audience', '')}\"
Формат результата: Строго 100-балльная шкала. Оценка ниже {user_input.get('success_threshold', 70)} считается негативным (неважным) примером.
Масштаб: \"{user_input.get('scale', '')}\"
Темы: {json.dumps(user_input.get('topics', []), ensure_ascii=False)}

Анализ примеров:
- Паттерны важных: {json.dumps(consolidation.get('positive_patterns', []), ensure_ascii=False)}
- Паттерны неважных: {json.dumps(consolidation.get('negative_patterns', []), ensure_ascii=False)}
- Ключевые отличия: {json.dumps(consolidation.get('key_differentiators', []), ensure_ascii=False)}
- Критерии: {json.dumps(consolidation.get('draft_criteria', []), ensure_ascii=False)}
- Слепые зоны: {json.dumps(consolidation.get('blind_spots', []), ensure_ascii=False)}{answers_text}

Размеченные примеры для калибровки:
{_format_examples_for_prompt(user_input.get('examples', []))}

Построй decision policy. Не пиши финальный промпт."""


# ── Step 4b: Evaluate prompt quality (100-point scale) ───────────

EVALUATE_SYSTEM = """Ты — строгий QA-инженер промптов. Оцени prompt только по тому, насколько он правильно разделяет размеченные пользователем примеры вокруг заданного порога и насколько его правила объясняют это разделение.

Если есть false positive, false negative или невалидный score, оценка separation_quality должна резко снижаться. Не ставь высокий total промпту, который ошибается на размеченных примерах.

Ответь СТРОГО в формате JSON без дополнительного текста."""

def _build_evaluate_user(prompt_text, test_results, examples, threshold, diagnostics):
    pos_results = []
    neg_results = []
    for r in test_results:
        entry = f"Документ: \"{r['text'][:200]}...\" → Оценка: {r.get('score', '?')}, Обоснование: \"{r.get('reasoning', '')[:200]}\""
        if r["label"] == "positive":
            pos_results.append(entry)
        else:
            neg_results.append(entry)

    return f"""Промпт: \"{prompt_text[:6000]}\"

Проходной балл, который установил пользователь: {threshold} (из 100).
ВНИМАНИЕ: Оценивай 'separation_quality' на основе этого порога! Если позитивный пример получил балл ниже {threshold}, или негативный пример получил балл выше или равный {threshold} - это грубая ошибка промпта, снижай баллы за разделение!

Автоматическая диагностика классификации:
{_diagnostics_text(diagnostics)}

Результаты тестового прогона:
=== ДОЛЖНЫ БЫТЬ ВЫСОКИЕ ОЦЕНКИ (позитивные примеры, ожидаем >= {threshold}) ===
{chr(10).join(pos_results) or 'Нет'}

=== ДОЛЖНЫ БЫТЬ НИЗКИЕ ОЦЕНКИ (негативные примеры, ожидаем < {threshold}) ===
{chr(10).join(neg_results) or 'Нет'}

Оцени в JSON:
{{
  "separation_quality": 0,
  "reasoning_quality": 0,
  "criteria_coverage": 0,
  "consistency": 0,
  "total": 0,
  "details": {{
    "separation_quality_comment": "Насколько чётко промпт разделяет позитивные и негативные примеры",
    "reasoning_quality_comment": "Насколько обоснования осмысленны",
    "criteria_coverage_comment": "Все ли темы и критерии учтены",
    "consistency_comment": "Стабильность оценок"
  }},
  "error_patterns": ["Какие типы документов промпт путает"],
  "scoring_adjustments": ["Какие веса или пороги надо изменить"],
  "prompt_patch_plan": ["Конкретная правка 1", "Конкретная правка 2"],
  "improvement_suggestions": ["Предложение 1", "Предложение 2"]
}}

Каждый критерий от 0 до 25. Итого — от 0 до 100."""


REPAIR_POLICY_SYSTEM = """Ты — архитектор decision policy. Улучши policy по результатам регрессионного теста.

Ты не переписываешь красивый промпт. Ты ремонтируешь правила принятия решения:
- false positive исправляй через hard reject, негативный сигнал, штраф, cap на максимальный score или конфликтное правило;
- false negative исправляй через новый/усиленный позитивный сигнал, изменение веса или уточнение hard reject, чтобы он не срабатывал слишком широко;
- если модель вернула невалидный score, уточни контракт JSON и шкалу;
- сохрани 100-балльную рубрику;
- не удаляй полезные правила без причины;
- в changes_made перечисли конкретные изменения.

Ответь СТРОГО JSON без дополнительного текста с теми же полями policy:
{
  "task_summary": "...",
  "important_definition": "...",
  "unimportant_definition": "...",
  "threshold": 70,
  "positive_signals": [{"name": "...", "description": "...", "weight": 25, "evidence": "..."}],
  "negative_signals": [{"name": "...", "description": "...", "penalty": -20, "evidence": "..."}],
  "hard_rejects": ["..."],
  "scoring_rubric": [{"criterion": "...", "weight": 25, "how_to_score": "..."}],
  "conflict_rules": ["..."],
  "edge_cases": [{"case": "...", "decision": "..."}],
  "calibration_rules": ["..."],
  "evidence_notes": ["..."],
  "changes_made": ["..."]
}"""

def _build_repair_policy_user(policy, prompt_text, evaluation, test_results, user_input, consolidation, answers, threshold):
    diagnostics = evaluation.get("diagnostics") or _build_test_diagnostics(test_results, threshold)
    answers_text = json.dumps(answers or {}, ensure_ascii=False)
    return f"""Текущая decision policy:
{json.dumps(policy, ensure_ascii=False, indent=2)[:10000]}

Скомпилированный промпт из этой policy:
\"{prompt_text[:5000]}\"

Score: {evaluation.get('total', 0)}/100

Проблемы: {json.dumps(evaluation.get('improvement_suggestions', []), ensure_ascii=False)}
Паттерны ошибок: {json.dumps(evaluation.get('error_patterns', []), ensure_ascii=False)}
План правок от оценщика: {json.dumps(evaluation.get('prompt_patch_plan', []), ensure_ascii=False)}
Нужные изменения скоринга: {json.dumps(evaluation.get('scoring_adjustments', []), ensure_ascii=False)}

Детали оценки:
- Разделение ({evaluation.get('separation_quality', 0)}/25): {evaluation.get('details', {}).get('separation_quality_comment', '')}
- Обоснования ({evaluation.get('reasoning_quality', 0)}/25): {evaluation.get('details', {}).get('reasoning_quality_comment', '')}
- Критерии ({evaluation.get('criteria_coverage', 0)}/25): {evaluation.get('details', {}).get('criteria_coverage_comment', '')}
- Стабильность ({evaluation.get('consistency', 0)}/25): {evaluation.get('details', {}).get('consistency_comment', '')}

Исходная задача пользователя: \"{user_input.get('user_comment', '')}\"
Целевая аудитория: \"{user_input.get('target_audience', '')}\"
Порог важности: {threshold}
Масштаб: \"{user_input.get('scale', '')}\"
Темы: {json.dumps(user_input.get('topics', []), ensure_ascii=False)}
Ответы на уточняющие вопросы: {answers_text}

Паттерны из анализа примеров:
- Важные: {json.dumps(consolidation.get('positive_patterns', []), ensure_ascii=False)}
- Неважные: {json.dumps(consolidation.get('negative_patterns', []), ensure_ascii=False)}
- Ключевые отличия: {json.dumps(consolidation.get('key_differentiators', []), ensure_ascii=False)}
- Критерии: {json.dumps(consolidation.get('draft_criteria', []), ensure_ascii=False)}

Диагностика ошибок, которые надо исправить:
{_diagnostics_text(diagnostics)}

Верни улучшенную decision policy. Не пиши финальный промпт."""


# ── State management ─────────────────────────────────────────────

def _new_lab_job(user_input):
    return {
        "status": "analyzing",
        "step": 2,
        "progress": {"current": 0, "total": 0, "stage": "Запуск анализа..."},
        "user_input": user_input,
        "analyses": [],          # Step 2a results
        "consolidation": None,   # Step 2b result
        "questions": None,       # Step 3 result
        "answers": None,         # Step 3 user answers
        "iterations": [],        # Step 4 iterations
        "error": None,
    }


def _update_lab_job(job_id, **updates):
    with LAB_LOCK:
        job = LAB_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)


def _update_lab_progress(job_id, current, total, stage):
    with LAB_LOCK:
        job = LAB_JOBS.get(job_id)
        if not job:
            return
        job["progress"] = {"current": current, "total": total, "stage": stage}


def lab_job_snapshot(job):
    """Build a JSON-serializable snapshot of the lab job."""
    iterations_clean = []
    for it in job.get("iterations", []):
        iterations_clean.append({
            "prompt_text": it.get("prompt_text", ""),
            "decision_policy": it.get("decision_policy", {}),
            "criteria_used": it.get("criteria_used", []),
            "explanation": it.get("explanation", ""),
            "test_results": [
                {
                    "text_preview": r.get("text", "")[:150],
                    "label": r.get("label", ""),
                    "score": r.get("score"),
                    "reasoning": r.get("reasoning", ""),
                }
                for r in it.get("test_results", [])
            ],
            "evaluation": it.get("evaluation", {}),
            "changes_made": it.get("changes_made", []),
        })

    analyses_clean = []
    for a in job.get("analyses", []):
        analyses_clean.append({
            "text_preview": a.get("text", "")[:150],
            "label": a.get("label", ""),
            "analysis": a.get("analysis", {}),
        })

    return {
        "status": job["status"],
        "step": job["step"],
        "progress": job["progress"],
        "analyses": analyses_clean,
        "consolidation": job.get("consolidation"),
        "questions": job.get("questions"),
        "iterations": iterations_clean,
        "error": job.get("error"),
    }


# ── Worker: Step 2 (analyze) ─────────────────────────────────────

def run_analysis_step(job_id):
    """Background worker: analyze examples + consolidate patterns."""
    try:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()

        with LAB_LOCK:
            job = LAB_JOBS.get(job_id)
            if not job:
                return
            examples = job["user_input"]["examples"]
            topics = job["user_input"].get("topics", [])
            user_input = job["user_input"]

        total = len(examples)
        analyses = []
        valid_examples = []

        # 2a: analyze each document individually
        for idx, example in enumerate(examples):
            with LAB_LOCK:
                if LAB_JOBS.get(job_id, {}).get("cancelled"):
                    return
            
            _update_lab_progress(job_id, idx + 1, total, f"Анализ документа {idx + 1} из {total}")

            task = ApiQueueTask(
                name=f"lab-analyze-{idx}",
                func=lambda t=example["text"], l=example["label"], tp=topics: _gc_call(
                    gc, ANALYZE_SINGLE_SYSTEM,
                    _build_analyze_single_user(t, l, tp)
                ),
            )
            API_QUEUE.submit(task)
            task.done.wait()

            analysis_result = task.result if not task.error else {"_error": str(task.error)}
            
            if "_error" in analysis_result:
                continue

            if analysis_result.get("is_broken"):
                continue
                
            valid_examples.append(example)

            analyses.append({
                "text": example["text"],
                "label": example["label"],
                "analysis": analysis_result,
            })

        with LAB_LOCK:
            if LAB_JOBS.get(job_id, {}).get("cancelled"):
                return
            LAB_JOBS[job_id]["analyses"] = analyses
            LAB_JOBS[job_id]["user_input"]["examples"] = valid_examples

        # 2b: consolidate
        _update_lab_progress(job_id, total, total, "Сведение паттернов...")

        task = ApiQueueTask(
            name="lab-consolidate",
            func=lambda: _gc_call(gc, CONSOLIDATE_SYSTEM, _build_consolidate_user(analyses, user_input)),
        )
        API_QUEUE.submit(task)
        task.done.wait()

        consolidation = task.result if not task.error else {"_error": str(task.error)}

        with LAB_LOCK:
            if LAB_JOBS.get(job_id, {}).get("cancelled"):
                return
            LAB_JOBS[job_id]["consolidation"] = consolidation

        # Step 3: questions (if enabled)
        ask_questions = user_input.get("ask_questions", True)

        if ask_questions:
            _update_lab_progress(job_id, total, total, "Генерация уточняющих вопросов...")

            task = ApiQueueTask(
                name="lab-questions",
                func=lambda: _gc_call(gc, QUESTIONS_SYSTEM, _build_questions_user(consolidation, user_input)),
            )
            API_QUEUE.submit(task)
            task.done.wait()

            questions = task.result if not task.error else None
            with LAB_LOCK:
                if LAB_JOBS.get(job_id, {}).get("cancelled"):
                    return
                LAB_JOBS[job_id]["questions"] = questions

            _update_lab_job(job_id, status="questions_ready", step=3,
                            progress={"current": total, "total": total, "stage": "Ожидание ответов на вопросы"})
        else:
            # Skip to generation
            _update_lab_job(job_id, status="generating", step=4)
            run_generation_step(job_id)

    except Exception as exc:
        traceback.print_exc()
        _update_lab_job(job_id, status="error", error=str(exc))


# ── Worker: Step 4 (generate + test + evaluate) ──────────────────

def run_generation_step(job_id, is_iteration=False):
    """Background worker: generate prompt, test it, evaluate, iterate."""
    try:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()

        with LAB_LOCK:
            job = LAB_JOBS.get(job_id)
            if not job:
                return
            consolidation = job["consolidation"] or {}
            user_input = job["user_input"]
            answers = job.get("answers")
            examples = user_input["examples"]
            iterations = job["iterations"]
            prev_prompt = iterations[-1]["prompt_text"] if iterations else None
            prev_policy = iterations[-1].get("decision_policy") if iterations else None

        total_examples = len(examples)
        threshold = user_input.get("success_threshold", 70)

        # 4a: build or repair decision policy, then compile prompt deterministically
        if is_iteration and prev_prompt and prev_policy:
            with LAB_LOCK:
                if LAB_JOBS.get(job_id, {}).get("cancelled"):
                    return
            prev_eval = iterations[-1].get("evaluation", {})
            prev_test_results = iterations[-1].get("test_results", [])
            _update_lab_progress(job_id, 0, total_examples, "Улучшение правил оценки...")

            task = ApiQueueTask(
                name="lab-repair-policy",
                func=lambda: _gc_call(
                    gc,
                    REPAIR_POLICY_SYSTEM,
                    _build_repair_policy_user(
                        prev_policy,
                        prev_prompt,
                        prev_eval,
                        prev_test_results,
                        user_input,
                        consolidation,
                        answers,
                        threshold,
                    ),
                ),
            )
            API_QUEUE.submit(task)
            task.done.wait()

            policy_result = task.result if not task.error else copy.deepcopy(prev_policy)
        else:
            with LAB_LOCK:
                if LAB_JOBS.get(job_id, {}).get("cancelled"):
                    return
            _update_lab_progress(job_id, 0, total_examples, "Построение правил оценки...")

            task = ApiQueueTask(
                name="lab-build-policy",
                func=lambda: _gc_call(gc, POLICY_SYSTEM, _build_policy_user(consolidation, user_input, answers)),
            )
            API_QUEUE.submit(task)
            task.done.wait()

            policy_result = task.result if not task.error else {"_error": str(task.error)}

        decision_policy = _normalize_policy(policy_result, user_input, consolidation, threshold)
        prompt_text = _compile_policy_to_prompt(decision_policy, user_input, threshold)
        if not prompt_text:
            _update_lab_job(job_id, status="error", error="AI не смог сгенерировать промпт")
            return

        # 4b: test prompt on all examples
        test_results = []
        for idx, example in enumerate(examples):
            with LAB_LOCK:
                if LAB_JOBS.get(job_id, {}).get("cancelled"):
                    return
            _update_lab_progress(job_id, idx + 1, total_examples,
                                 f"Тестовый прогон {idx + 1} из {total_examples}")

            task = ApiQueueTask(
                name=f"lab-test-{idx}",
                func=lambda t=example["text"], p=prompt_text: _gc_call(gc, p, t),
            )
            API_QUEUE.submit(task)
            task.done.wait()

            result = task.result if not task.error else {"_error": str(task.error)}
            
            if "_error" in result:
                continue

            test_results.append({
                "text": example["text"],
                "label": example["label"],
                "score": result.get("score"),
                "reasoning": result.get("reasoning", result.get("_raw", "")),
            })

        # 4c: evaluate
        with LAB_LOCK:
            if LAB_JOBS.get(job_id, {}).get("cancelled"):
                return
        _update_lab_progress(job_id, total_examples, total_examples, "Оценка качества промпта...")

        diagnostics = _build_test_diagnostics(test_results, threshold)

        task = ApiQueueTask(
            name="lab-evaluate",
            func=lambda: _gc_call(gc, EVALUATE_SYSTEM,
                                   _build_evaluate_user(prompt_text, test_results, examples, threshold, diagnostics)),
        )
        API_QUEUE.submit(task)
        task.done.wait()

        evaluation = task.result if not task.error else {"total": 0, "_error": str(task.error)}
        evaluation = _calibrate_evaluation(evaluation, diagnostics)

        # Save iteration
        iteration_data = {
            "prompt_text": prompt_text,
            "decision_policy": decision_policy,
            "criteria_used": decision_policy.get("scoring_rubric", []),
            "explanation": "Промпт скомпилирован из decision policy: правила, веса, исключения и калибровка по примерам.",
            "changes_made": decision_policy.get("changes_made", []),
            "test_results": test_results,
            "evaluation": evaluation,
        }

        with LAB_LOCK:
            LAB_JOBS[job_id]["iterations"].append(iteration_data)

        total_score = evaluation.get("total", 0)
        iteration_count = len(LAB_JOBS[job_id]["iterations"])

        # Auto-iterate if score < 90 and < 4 iterations
        if total_score < 90 and iteration_count < 4:
            _update_lab_job(job_id, status="generating", step=4)
            run_generation_step(job_id, is_iteration=True)
        else:
            stage = f"Готово! Финальный Score: {total_score}/100 ({iteration_count} итераций)"
            _update_lab_job(
                job_id, status="done", step=4,
                progress={"current": total_examples, "total": total_examples, "stage": stage},
            )

    except Exception as exc:
        traceback.print_exc()
        _update_lab_job(job_id, status="error", error=str(exc))
