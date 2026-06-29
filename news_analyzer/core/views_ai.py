"""
HTTP-эндпоинты для двух AI-сценариев на эмбеддингах:

  • Обозреватель (А) — семантический few-shot отбор + проверка на эталоне.
    review/run/      — запустить оценку (leave-one-out по колонке-эталону)
    review/progress/ — статус + метрики (precision/recall, подбор порога)

  • Гипотезы (Б) — генерация аудиторских гипотез по теме риска.
    hypo/start/      — запустить (file_path, sheet, theme, top_n)
    hypo/progress/   — статус + список гипотез
    hypo/cancel/     — отменить

Воркеры крутятся в daemon-потоках с in-memory состоянием (как в prompt_lab).
Эталонная колонка «Статус отбора» опциональна: review/ требует её, hypo/ — нет.
"""

import json
import threading
import traceback
import uuid

import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .features import resolve_columns, get_content, get
from .evaluation import is_positive_label, metrics_at_threshold, recommend_threshold, sweep_thresholds


# ══════════════════════════════════════════════════════════════════
# Обозреватель: проверка на эталоне (leave-one-out)
# ══════════════════════════════════════════════════════════════════
REVIEW_JOBS = {}
REVIEW_LOCK = threading.Lock()


def _review_set(job_id, **updates):
    with REVIEW_LOCK:
        job = REVIEW_JOBS.get(job_id)
        if job:
            job.update(updates)


def _run_review_eval(job_id, file_path, sheet, k, threshold):
    try:
        from .gigachat import GigaChatClient
        from .reviewer import build_index_from_labeled, REVIEW_SYSTEM, build_review_user, _parse_review
        from .embeddings import embed_texts

        gc = GigaChatClient.get_instance()
        df = pd.read_excel(file_path, sheet_name=sheet)
        colmap = resolve_columns(df.columns)
        if "label" not in colmap:
            _review_set(job_id, status="error", error="Нет колонки-эталона отбора (напр. «Статус отбора»)")
            return

        rows = df.to_dict("records")
        labeled = [
            (idx, get_content(rows[idx], colmap), is_positive_label(get(rows[idx], colmap, "label", "")))
            for idx in range(len(rows))
            if get_content(rows[idx], colmap).strip()
        ]
        total = len(labeled)
        _review_set(job_id, total=total, stage="Эмбеддинг примеров...")

        # один индекс на всех; при оценке строки исключаем её саму (leave-one-out)
        index = build_index_from_labeled(rows, colmap, gc=gc)

        pairs = []
        details = []
        for n, (idx, content, is_pos) in enumerate(labeled, 1):
            with REVIEW_LOCK:
                if REVIEW_JOBS.get(job_id, {}).get("cancelled"):
                    return
            _review_set(job_id, current=n, stage=f"Оценка {n} из {total}")
            qv = embed_texts([content], gc=gc)[0]
            pos = index.search(query_vec=qv, k=k, where=lambda p: p["is_positive"] and p["row_index"] != idx)
            neg = index.search(query_vec=qv, k=k, where=lambda p: not p["is_positive"] and p["row_index"] != idx)
            resp = gc.completions(
                [{"role": "system", "content": REVIEW_SYSTEM},
                 {"role": "user", "content": build_review_user(content, pos, neg, threshold)}],
                temperature=0,
            )
            score, reasoning = _parse_review(resp["choices"][0]["message"]["content"].strip())
            pairs.append((is_pos, score))
            details.append({"row_index": idx, "is_positive": is_pos, "score": score, "reasoning": reasoning[:300]})

        metrics = metrics_at_threshold(pairs, threshold)
        recommended = recommend_threshold(pairs, min_recall=1.0)
        sweep = sweep_thresholds(pairs, 0, 100, 10)
        _review_set(
            job_id, status="done", stage="Готово",
            metrics=metrics, recommended=recommended, sweep=sweep, details=details,
        )
    except Exception as exc:
        traceback.print_exc()
        _review_set(job_id, status="error", error=str(exc))


@csrf_exempt
def review_run(request):
    if request.method != "POST":
        return JsonResponse({"error": "Только POST"}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)

    file_path = data.get("file_path", "")
    sheet = data.get("sheet", "")
    k = int(data.get("k", 3))
    threshold = float(data.get("threshold", 70))
    if not file_path or not sheet:
        return JsonResponse({"error": "Не указан файл или лист"}, status=400)

    job_id = str(uuid.uuid4())
    with REVIEW_LOCK:
        REVIEW_JOBS[job_id] = {
            "status": "running", "stage": "Запуск...", "current": 0, "total": 0,
            "metrics": None, "recommended": None, "sweep": None, "details": [],
            "error": None, "cancelled": False,
        }
    threading.Thread(target=_run_review_eval, args=(job_id, file_path, sheet, k, threshold), daemon=True).start()
    return JsonResponse({"job_id": job_id})


def review_progress(request):
    job_id = request.GET.get("job_id", "")
    with REVIEW_LOCK:
        job = REVIEW_JOBS.get(job_id)
        if not job:
            return JsonResponse({"error": "Job not found"}, status=404)
        snapshot = {k: v for k, v in job.items() if k != "cancelled"}
    return JsonResponse(snapshot)


@csrf_exempt
def review_cancel(request):
    try:
        data = json.loads(request.body)
        with REVIEW_LOCK:
            job = REVIEW_JOBS.get(data.get("job_id"))
            if job:
                job["cancelled"] = True
                job["status"] = "error"
                job["error"] = "Отменено пользователем"
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"ok": True})


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

    df = pd.read_excel(file_path, sheet_name=sheet)
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
