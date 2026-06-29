"""
Обезличивание (де-идентификация) текста — удаление персональных данных.

Это ЗАЩИТНЫЙ фильтр, а не аналитика, поэтому для строго форматных ПДн
(телефон, паспорт, СНИЛС, e-mail, карта, ИНН) используется regex — он
надёжнее слабого LLM и не «пропускает» номер. ФИО ловятся паттерном
«Фамилия И.О.» (в документах СберДокс почти всегда так), а опциональный
LLM-проход добирает полные имена, которые regex не покрыл.

Каждый тип ПДн заменяется своим плейсхолдером: <name>, <phone>, <email>,
<passport>, <snils>, <inn>, <card>.
"""

import re


# Порядок важен: длинные/специфичные форматы раньше общих числовых.
_PATTERNS = [
    ("email", "<email>", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("card", "<card>", re.compile(r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b")),
    ("snils", "<snils>", re.compile(r"\b\d{3}-\d{3}-\d{3}[ ]?\d{2}\b")),
    # телефон РФ: +7/8 и 10 цифр в разных разделителях
    ("phone", "<phone>", re.compile(
        r"(?:\+7|\b8|\b7)[ \-]?\(?\d{3}\)?[ \-]?\d{3}[ \-]?\d{2}[ \-]?\d{2}\b")),
    # ИНН — по ключевому слову, ДО паспорта (10 цифр иначе перехватятся паспортом)
    ("inn", "<inn>", re.compile(r"\bИНН[:\s]*\d{10,12}\b", re.IGNORECASE)),
    # паспорт РФ: серия (4) + номер (6), допускаем пробелы
    ("passport", "<passport>", re.compile(r"\b\d{2}[ ]?\d{2}[ ]?\d{6}\b")),
    # личный ИНН — 12 цифр подряд (запасной отлов без ключевого слова)
    ("inn", "<inn>", re.compile(r"\b\d{12}\b")),
]

# ФИО с инициалами: «Греф Г.О.», «Ведяхин А.А.» и обратный порядок «Г.О. Греф»
_FIO_SURNAME_INITIALS = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.[ ]?[А-ЯЁ]\.")
_FIO_INITIALS_SURNAME = re.compile(r"\b[А-ЯЁ]\.[ ]?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")


def anonymize_text(text, use_llm=False, gc=None):
    """
    Вернуть (обезличенный_текст, статистика_по_типам).
    use_llm=True добавляет проход GigaChat для полных ФИО без инициалов.
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

    for label, placeholder, pattern in _PATTERNS:
        result = _sub(label, placeholder, pattern, result)

    # ФИО по инициалам (оба порядка)
    for pattern in (_FIO_INITIALS_SURNAME, _FIO_SURNAME_INITIALS):
        result = _sub("name", "<name>", pattern, result)

    # Опциональный LLM-проход для полных имён
    if use_llm:
        names = _llm_find_names(result, gc)
        for name in names:
            if name and name in result:
                stats["name"] = stats.get("name", 0) + result.count(name)
                result = result.replace(name, "<name>")

    return result, stats


_LLM_NER_SYSTEM = """Ты — фильтр персональных данных. Найди в тексте ВСЕ упоминания ФИО людей
(имена, фамилии, отчества в любом виде). Не включай названия организаций, городов, должностей.
Отвечай СТРОГО JSON-массивом строк, например: ["Иван Петров", "Мария С. Иванова"]. Если имён нет — []."""


def _llm_find_names(text, gc=None):
    import json
    if gc is None:
        from .gigachat import GigaChatClient
        gc = GigaChatClient.get_instance()
    try:
        resp = gc.completions(
            [{"role": "system", "content": _LLM_NER_SYSTEM},
             {"role": "user", "content": text[:8000]}],
            temperature=0,
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data = json.loads(raw)
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []
