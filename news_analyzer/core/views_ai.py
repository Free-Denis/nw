"""
HTTP-эндпоинты AI-сценариев на эмбеддингах:

  • Гипотезы (Б) — генерация аудиторских гипотез по теме риска.
    hypo/start/    — запустить (file_path, sheet, theme, top_n)
    hypo/progress/ — статус + список гипотез
    hypo/cancel/   — отменить

  • Обезличивание — удаление персональных данных из текста.
    anon/run/      — синхронно обезличить текст

Воркеры крутятся в daemon-потоках с in-memory состоянием (как в prompt_lab).
"""

import json
import threading
import traceback
import uuid
from pathlib import Path as PyPath

import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .features import resolve_columns

ANON_DIR = PyPath("storage/anon")
ANON_DIR.mkdir(parents=True, exist_ok=True)


def _active_text_model():
    from .views import _load_settings, DEFAULT_TEXT_MODEL
    return _load_settings().get("active_text_model", DEFAULT_TEXT_MODEL)


# ══════════════════════════════════════════════════════════════════
# Гипотезы
# ══════════════════════════════════════════════════════════════════

@csrf_exempt
def hypo_start(request):
    from .hypotheses import HYPO_JOBS, HYPO_LOCK, _new_hypo_job, run_hypothesis_job

    if request.method != "POST":
        return JsonResponse({"error": "Только POST"}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)

    file_path = data.get("file_path", "")
    sheet = data.get("sheet", "")
    if not file_path or not sheet:
        return JsonResponse({"error": "Не указан файл или лист"}, status=400)

    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except Exception as exc:
        return JsonResponse({"error": f"Не удалось прочитать файл: {exc}"}, status=500)
    colmap = resolve_columns(df.columns)
    rows = df.to_dict("records")
    model = _active_text_model()

    job_id = str(uuid.uuid4())
    with HYPO_LOCK:
        HYPO_JOBS[job_id] = _new_hypo_job()
    threading.Thread(
        target=run_hypothesis_job, args=(job_id, rows, colmap, model), daemon=True
    ).start()
    return JsonResponse({"job_id": job_id})


def hypo_progress(request):
    from .hypotheses import HYPO_JOBS, HYPO_LOCK

    job_id = request.GET.get("job_id", "")
    with HYPO_LOCK:
        job = HYPO_JOBS.get(job_id)
        if not job:
            return JsonResponse({"error": "Job not found"}, status=404)
        snapshot = {
            "status": job["status"],
            "progress": job["progress"],
            "hypotheses": job["hypotheses"],
            "groups": job["groups"],
            "error": job["error"],
        }
    return JsonResponse(snapshot)


@csrf_exempt
def hypo_cancel(request):
    from .hypotheses import HYPO_JOBS, HYPO_LOCK

    try:
        data = json.loads(request.body)
        with HYPO_LOCK:
            job = HYPO_JOBS.get(data.get("job_id"))
            if job:
                job["cancelled"] = True
                job["status"] = "error"
                job["error"] = "Отменено пользователем"
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"ok": True})


# ══════════════════════════════════════════════════════════════════
# Обезличивание
# ══════════════════════════════════════════════════════════════════

@csrf_exempt
def anon_run(request):
    """Синхронно обезличить переданный текст. body: {text, use_llm}."""
    from .anonymize import anonymize_text

    if request.method != "POST":
        return JsonResponse({"error": "Только POST"}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)

    text = str(data.get("text", ""))
    if not text.strip():
        return JsonResponse({"error": "Пустой текст"}, status=400)
    use_llm = bool(data.get("use_llm", True))
    try:
        clean, stats = anonymize_text(text, use_llm=use_llm, model=_active_text_model())
        return JsonResponse({"clean_text": clean, "stats": stats})
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({"error": str(exc)}, status=500)


# ══════════════════════════════════════════════════════════════════
# Обезличивание Excel-столбца (пакетно, в фоне)
# ══════════════════════════════════════════════════════════════════
ANON_JOBS = {}
ANON_LOCK = threading.Lock()


def _anon_set(job_id, **updates):
    with ANON_LOCK:
        job = ANON_JOBS.get(job_id)
        if job:
            job.update(updates)


def _run_anon_excel(job_id, file_path, sheet, column, use_llm):
    from .anonymize import anonymize_text
    try:
        model = _active_text_model()
        df = pd.read_excel(file_path, sheet_name=sheet)
        if column not in df.columns:
            _anon_set(job_id, status="error", error=f"Столбец «{column}» не найден")
            return

        out_col = f"{column} (обезличено)"
        values = df[column].tolist()
        total = len(values)
        _anon_set(job_id, total=total, stage="Обезличивание...")

        cleaned = []
        for i, val in enumerate(values, 1):
            with ANON_LOCK:
                if ANON_JOBS.get(job_id, {}).get("cancelled"):
                    return
            text = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)
            clean, _stats = anonymize_text(text, use_llm=use_llm, model=model) if text.strip() else ("", {})
            cleaned.append(clean)
            _anon_set(job_id, current=i, stage=f"Обезличивание {i} из {total}")

        df[out_col] = cleaned
        out_path = ANON_DIR / f"anon_{uuid.uuid4().hex[:8]}.xlsx"
        df.to_excel(out_path, sheet_name=str(sheet)[:31], index=False)
        _anon_set(
            job_id, status="done", stage="Готово",
            download_url=f"download_rated/?path={out_path.absolute()}",
        )
    except Exception as exc:
        traceback.print_exc()
        _anon_set(job_id, status="error", error=str(exc))


@csrf_exempt
def anon_excel_start(request):
    if request.method != "POST":
        return JsonResponse({"error": "Только POST"}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)

    file_path = data.get("file_path", "")
    sheet = data.get("sheet", "")
    column = data.get("column", "")
    use_llm = bool(data.get("use_llm", True))
    if not file_path or not sheet or not column:
        return JsonResponse({"error": "Укажите файл, лист и столбец"}, status=400)

    job_id = str(uuid.uuid4())
    with ANON_LOCK:
        ANON_JOBS[job_id] = {
            "status": "running", "stage": "Запуск...", "current": 0, "total": 0,
            "download_url": None, "error": None, "cancelled": False,
        }
    threading.Thread(target=_run_anon_excel, args=(job_id, file_path, sheet, column, use_llm), daemon=True).start()
    return JsonResponse({"job_id": job_id})


def anon_excel_progress(request):
    job_id = request.GET.get("job_id", "")
    with ANON_LOCK:
        job = ANON_JOBS.get(job_id)
        if not job:
            return JsonResponse({"error": "Job not found"}, status=404)
        snapshot = {k: v for k, v in job.items() if k != "cancelled"}
    return JsonResponse(snapshot)


@csrf_exempt
def anon_excel_cancel(request):
    try:
        data = json.loads(request.body)
        with ANON_LOCK:
            job = ANON_JOBS.get(data.get("job_id"))
            if job:
                job["cancelled"] = True
                job["status"] = "error"
                job["error"] = "Отменено пользователем"
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"ok": True})
