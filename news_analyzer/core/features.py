"""
Document text access — устойчивое чтение нужных колонок из выгрузки СберДокс.

По решению эксперта метаданные (отправитель, вид документа, кол-во получателей
и т.п.) в анализе НЕ используются — опираемся на содержание. Этот модуль лишь
помогает достать чистый текст документа из выгрузки, колонки которой могут
называться по-разному.
"""

import pandas as pd


# Канонические имена → варианты названий колонок (поиск по подстроке).
COLUMN_ALIASES = {
    "subject": ["содержание", "тема", "краткое содержание"],
    "text": ["текст", "полный текст", "содержимое"],
    "link": ["ссылка", "url"],
    # eval-only, может отсутствовать в боевой выгрузке
    "label": ["статус отбора", "отбор", "метка"],
}


def resolve_columns(columns):
    """Вернуть {канон. имя -> реальное имя колонки} по подстроке."""
    lowered = {str(c).strip().lower(): c for c in columns}
    resolved = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            match = next((orig for low, orig in lowered.items() if alias in low), None)
            if match is not None:
                resolved[canonical] = match
                break
    return resolved


def get(row, colmap, canonical, default=""):
    """Безопасно достать значение по каноническому имени колонки."""
    col = colmap.get(canonical)
    if col is None:
        return default
    value = row.get(col)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return default if stripped.lower() == "nan" else stripped
    return value


def get_content(row, colmap):
    """Содержательный текст документа для анализа: «Содержание» + «Текст»."""
    subject = str(get(row, colmap, "subject", "")).strip()
    text = str(get(row, colmap, "text", "")).strip()
    if subject and text:
        return f"{subject}\n\n{text}"
    return subject or text
