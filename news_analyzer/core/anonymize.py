"""
Обезличивание (де-идентификация) текста — удаление персональных данных.

Стратегия:
  • Форматные ПДн (телефон, паспорт, СНИЛС, e-mail, карта, ИНН) — regex.
    Они строго форматные, regex надёжнее и не «пропускает» номер.
  • ФИО — нейросетью (glm-5.1): модель возвращает имена РОВНО как в тексте,
    в том же падеже, поэтому склоняемые формы («Дерябину Денису Алексеевичу»)
    заменяются буквально. Regex «Фамилия И.О.» оставлен как быстрый
    дополнительный проход для частого корпоративного формата.

Каждый тип ПДн заменяется своим плейсхолдером: <name>, <phone>, <email>,
<passport>, <snils>, <inn>, <card>.
"""

import json
import re


# Порядок важен: длинные/специфичные форматы раньше общих числовых.
_PATTERNS = [
    ("email", "<email>", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("card", "<card>", re.compile(r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b")),
    ("snils", "<snils>", re.compile(r"\b\d{3}-\d{3}-\d{3}[ ]?\d{2}\b")),
    ("phone", "<phone>", re.compile(
        r"(?:\+7|\b8|\b7)[ \-]?\(?\d{3}\)?[ \-]?\d{3}[ \-]?\d{2}[ \-]?\d{2}\b")),
    ("inn", "<inn>", re.compile(r"\bИНН[:\s]*\d{10,12}\b", re.IGNORECASE)),
    ("passport", "<passport>", re.compile(r"\b\d{2}[ ]?\d{2}[ ]?\d{6}\b")),
    ("inn", "<inn>", re.compile(r"\b\d{12}\b")),
]

# ФИО с инициалами: «Греф Г.О.», «Ведяхин А.А.» и обратный порядок «Г.О. Греф»
_FIO_SURNAME_INITIALS = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.[ ]?[А-ЯЁ]\.")
_FIO_INITIALS_SURNAME = re.compile(r"\b[А-ЯЁ]\.[ ]?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")


def anonymize_text(text, use_llm=True, gc=None, model=None):
    """
    Вернуть (обезличенный_текст, статистика_по_типам).
    use_llm=True (по умолчанию) включает нейросетевой поиск ФИО — основной путь.
    """
    if not text:
        return "", {}

    stats = {}
    result = text

    def _sub(label, placeholder, pattern, s):
        count = len(pattern.findall(s))
        if count:
            stats[label] = stats.get(label, 0) + count
        return pattern.sub(placeholder, s)

    # 1) форматные ПДн — regex
    for label, placeholder, pattern in _PATTERNS:
        result = _sub(label, placeholder, pattern, result)

    # 2) ФИО по инициалам (быстрый проход для «Фамилия И.О.»)
    for pattern in (_FIO_INITIALS_SURNAME, _FIO_SURNAME_INITIALS):
        result = _sub("name", "<name>", pattern, result)

    # 3) ФИО нейросетью — ловит полные склоняемые имена
    if use_llm:
        names = _llm_find_names(result, gc=gc, model=model)
        # сначала длинные строки, чтобы не разрезать «Денису Алексеевичу» на части
        for name in sorted(set(names), key=len, reverse=True):
            if _is_replaceable_name(name) and name in result:
                stats["name"] = stats.get("name", 0) + result.count(name)
                result = result.replace(name, "<name>")

    return result, stats


def _is_replaceable_name(name):
    """Отсечь мусор: слишком короткое, без заглавной кириллицы, плейсхолдер."""
    name = (name or "").strip()
    if len(name) < 4 or name.startswith("<"):
        return False
    return bool(re.search(r"[А-ЯЁ]", name))


_LLM_NER_SYSTEM = """Ты — фильтр персональных данных. Найди в тексте ВСЕ упоминания ФИО людей
(фамилии, имена, отчества и их сочетания) в ЛЮБОМ падеже и форме.

ВАЖНО:
- Выписывай каждое упоминание РОВНО как оно встречается в тексте (тот же падеж, та же форма).
  Пример: если в тексте «направить Дерябину Денису Алексеевичу», верни "Дерябину Денису Алексеевичу".
- Каждое отдельное вхождение — отдельный элемент массива (включая повторы в разных падежах).
- НЕ включай названия организаций, городов, должностей, продуктов.

Ответь СТРОГО JSON-массивом строк, например: ["Дерябину Денису Алексеевичу", "Греф Г.О."].
Если имён нет — верни []."""


def _llm_find_names(text, gc=None, model=None):
    if gc is None:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()
    try:
        resp = gc.completions(
            [{"role": "system", "content": _LLM_NER_SYSTEM},
             {"role": "user", "content": text[:8000]}],
            temperature=0,
            model=model,
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data = json.loads(raw)
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []
