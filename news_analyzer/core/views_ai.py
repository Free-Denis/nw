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

import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .features import resolve_columns


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
    theme = str(data.get("theme", "")).strip()
    top_n = int(data.get("top_n", 15))
    if not file_path or not sheet:
        return JsonResponse({"error": "Не указан файл или лист"}, status=400)
    if len(theme) < 3:
        return JsonResponse({"error": "Укажите тему риска (напр. «мошенничество», «налоги»)"}, status=400)

    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except Exception as exc:
        return JsonResponse({"error": f"Не удалось прочитать файл: {exc}"}, status=500)
    colmap = resolve_columns(df.columns)
    rows = df.to_dict("records")

    job_id = str(uuid.uuid4())
    with HYPO_LOCK:
        HYPO_JOBS[job_id] = _new_hypo_job(theme, top_n)
    threading.Thread(
        target=run_hypothesis_job, args=(job_id, rows, colmap, theme, top_n), daemon=True
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
            "theme": job["theme"],
            "progress": job["progress"],
            "hypotheses": job["hypotheses"],
            "clusters": job["clusters"],
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
    use_llm = bool(data.get("use_llm", False))
    try:
        clean, stats = anonymize_text(text, use_llm=use_llm)
        return JsonResponse({"clean_text": clean, "stats": stats})
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({"error": str(exc)}, status=500)
