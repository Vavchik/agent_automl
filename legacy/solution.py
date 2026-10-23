"""Точка входа платформы: python solution.py task.md

Р1. Здесь только страховка: любой сбой пайплайна не должен оставить нас без submission.csv.
Вся логика — в core/pipeline.py.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.fallback import write_fallback  # noqa: E402  (импорт без тяжёлых зависимостей)


def main() -> None:
    task_md = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    try:
        from core.pipeline import run
        run(task_md)
    except BaseException:
        traceback.print_exc()
        if not (task_md.parent / "submission.csv").exists():
            path = write_fallback(task_md)
            print(f"Пайплайн упал до первого сабмита, записан аварийный ответ: {path}")


if __name__ == "__main__":
    main()
