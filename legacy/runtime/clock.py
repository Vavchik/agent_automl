"""Р1. Общий дедлайн запуска. Каждая стадия спрашивает у часов, успеет ли она."""
import time

import config


class Clock:
    def __init__(self, limit_sec: float = config.TIME_LIMIT_SEC, margin_sec: float = config.SAFETY_MARGIN_SEC):
        self.start = time.time()
        self.limit_sec = limit_sec
        self.margin_sec = margin_sec

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        """Сколько секунд ещё можно работать с учётом запаса на финал."""
        return self.limit_sec - self.margin_sec - self.elapsed()

    def has(self, seconds: float) -> bool:
        return self.remaining() > seconds
