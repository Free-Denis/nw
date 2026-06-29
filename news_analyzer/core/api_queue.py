from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import threading
import time
import traceback
import uuid


@dataclass
class ApiQueueTask:
    name: str
    func: Callable[[], Any]
    priority: bool = False
    max_retries: int = 3
    retry_delay: int = 5
    on_attempt_error: Optional[Callable[[Exception, int, int], None]] = None
    on_success: Optional[Callable[[Any], None]] = None
    on_failure: Optional[Callable[[Exception], None]] = None
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[Exception] = None


class ApiQueueManager:
    def __init__(self):
        self._queue = deque()
        self._condition = threading.Condition()
        self._active_task_id = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, task: ApiQueueTask) -> ApiQueueTask:
        with self._condition:
            if task.priority:
                self._queue.appendleft(task)
            else:
                self._queue.append(task)
            self._condition.notify()
        return task

    def has_pending_work(self) -> bool:
        with self._condition:
            return bool(self._queue) or self._active_task_id is not None

    def clear(self):
        with self._condition:
            for task in self._queue:
                task.error = Exception("Очередь очищена (загружен новый файл)")
                task.done.set()
            self._queue.clear()

    def _worker_loop(self):
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                task = self._queue.popleft()
                self._active_task_id = task.task_id

            try:
                self._execute_task(task)
            finally:
                with self._condition:
                    self._active_task_id = None
                task.done.set()

    def _execute_task(self, task: ApiQueueTask):
        last_error = None

        for attempt in range(1, task.max_retries + 1):
            try:
                result = task.func()
                task.result = result
                if task.on_success:
                    task.on_success(result)
                return
            except Exception as exc:
                last_error = exc
                if attempt < task.max_retries:
                    if task.on_attempt_error:
                        try:
                            task.on_attempt_error(exc, attempt, task.max_retries)
                        except Exception:
                            traceback.print_exc()
                    time.sleep(task.retry_delay)
                else:
                    task.error = exc
                    if task.on_failure:
                        try:
                            task.on_failure(exc)
                        except Exception:
                            traceback.print_exc()

        if last_error:
            task.error = last_error


API_QUEUE = ApiQueueManager()
