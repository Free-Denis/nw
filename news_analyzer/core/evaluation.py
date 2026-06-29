"""
Evaluation / metrics — pure code.

Когда в выгрузке есть колонка-эталон («Статус отбора» = выбран/не выбран),
этот модуль считает, насколько оценки инструмента совпадают с выбором эксперта:
precision / recall / F1 при заданном пороге и подбор оптимального порога.

В боевой выгрузке колонки-эталона нет — тогда метрики просто недоступны,
а инструмент работает по откалиброванному ранее порогу.
"""

POSITIVE_LABELS = {"выбран", "выбрано", "да", "1", "true", "important", "важно"}


def is_positive_label(value):
    return str(value).strip().lower() in POSITIVE_LABELS


def confusion_at_threshold(pairs, threshold):
    """pairs = [(is_positive: bool, score: float|None)]. Вернуть TP/FP/FN/TN."""
    tp = fp = fn = tn = 0
    for is_pos, score in pairs:
        predicted_pos = score is not None and score >= threshold
        if is_pos and predicted_pos:
            tp += 1
        elif is_pos and not predicted_pos:
            fn += 1
        elif not is_pos and predicted_pos:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_at_threshold(pairs, threshold):
    c = confusion_at_threshold(pairs, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "threshold": threshold,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        **c,
        "positives": tp + fn,
        "predicted_positive": tp + fp,
    }


def sweep_thresholds(pairs, start=0, stop=100, step=5):
    """Метрики по сетке порогов."""
    return [metrics_at_threshold(pairs, t) for t in range(start, stop + 1, step)]


def recommend_threshold(pairs, min_recall=1.0, start=0, stop=100, step=5):
    """
    Подобрать порог под политику аудита: сначала добиться recall >= min_recall
    (не пропустить нужные эксперту документы), среди таких — максимум precision.
    Если recall недостижим — вернуть порог с максимальным F1.
    """
    grid = sweep_thresholds(pairs, start, stop, step)
    eligible = [m for m in grid if m["recall"] >= min_recall]
    if eligible:
        best = max(eligible, key=lambda m: (m["precision"], m["threshold"]))
        best = dict(best)
        best["rationale"] = f"Recall ≥ {min_recall:.0%} достигнут, максимизирована precision"
        return best
    best = max(grid, key=lambda m: (m["f1"], m["recall"]))
    best = dict(best)
    best["rationale"] = f"Recall {min_recall:.0%} недостижим, выбран максимум F1"
    return best


def format_report(metrics):
    """Человекочитаемый отчёт по одному порогу."""
    return (
        f"Порог: {metrics['threshold']}\n"
        f"Поймано эксперта: {metrics['tp']} из {metrics['positives']} "
        f"(recall {metrics['recall']:.0%})\n"
        f"Ложных срабатываний: {metrics['fp']} "
        f"(precision {metrics['precision']:.0%})\n"
        f"Пропущено: {metrics['fn']}\n"
        f"F1: {metrics['f1']:.2f}"
    )
