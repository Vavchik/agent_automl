#!/usr/bin/env python3
"""Р1. Сборка solution.zip под регламент Boosters: в корне архива только solution.py и requirements.txt.

Как работает: все модули агента (config.py, core, runtime, task, ml, agent, validation) упаковываются
в zip, кодируются в base64 и вкладываются строкой в единый dist/solution.py. При запуске на платформе
этот файл распаковывает модули во временную папку и вызывает core.pipeline.run — тот же код,
что в репозитории. Ничего не копируется руками, поэтому собранная версия не расходится с исходниками.

Запуск: python3 scripts/bundle_solution.py   (или scripts/build_zip.sh)
"""
from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["core", "runtime", "task", "ml", "agent", "validation"]
SINGLE_FILES = ["config.py"]

TEMPLATE = '''"""Единая точка входа для платформы Boosters (Альфа Хак: AutoML): python solution.py task.md

Файл собран scripts/bundle_solution.py из модулей репозитория. Не редактируйте его вручную:
правьте исходники и пересоберите scripts/build_zip.sh.
"""
import base64
import io
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

_BUNDLE = "{bundle}"


def _unpack() -> Path:
    target = Path(tempfile.mkdtemp(prefix="automl_agent_"))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_BUNDLE))) as archive:
        archive.extractall(target)
    sys.path.insert(0, str(target))
    return target


def _emergency(task_md: Path) -> None:
    """Последний рубеж, если не сработал даже runtime.fallback: копия sample_submission."""
    task_dir = task_md.parent
    found = sorted(task_dir.rglob("sample_submission.csv"))
    if found:
        (task_dir / "submission.csv").write_bytes(found[0].read_bytes())


def main() -> None:
    task_md = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    try:
        _unpack()
        from core.pipeline import run
        run(task_md)
    except BaseException:
        traceback.print_exc()
        if not (task_md.parent / "submission.csv").exists():
            try:
                from runtime.fallback import write_fallback
                print(f"Пайплайн упал до первого сабмита, аварийный ответ: {{write_fallback(task_md)}}")
            except BaseException:
                traceback.print_exc()
                _emergency(task_md)


if __name__ == "__main__":
    main()
'''


def collect_sources() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for package in PACKAGES:
            for path in sorted((ROOT / package).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    archive.write(path, arcname=str(path.relative_to(ROOT)))
        for name in SINGLE_FILES:
            archive.write(ROOT / name, arcname=name)
    return buffer.getvalue()


def main() -> None:
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    payload = base64.b64encode(collect_sources()).decode("ascii")
    solution = dist / "solution.py"
    solution.write_text(TEMPLATE.format(bundle=payload), encoding="utf-8")
    compile(solution.read_text(encoding="utf-8"), str(solution), "exec")   # проверка синтаксиса
    (dist / "requirements.txt").write_bytes((ROOT / "requirements.txt").read_bytes())

    out_zip = ROOT / "solution.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(solution, arcname="solution.py")
        archive.write(dist / "requirements.txt", arcname="requirements.txt")
    print(f"Собран {solution} ({solution.stat().st_size // 1024} КБ) и {out_zip}")


if __name__ == "__main__":
    main()
