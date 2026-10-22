import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int = 0
    duration: float = 0.0
    script_path: str = ""

def execute_code(
    code_str: str,
    timeout: int = 600,
    cwd: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> ExecutionResult:
    """
    Безопасно сохраняет и запускает переданный Python-код в отдельном процессе.
    
    Args:
        code_str: Строка с кодом, который написал LLM-агент.
        timeout: Максимальное время выполнения в секундах (по умолчанию 600 с).
        cwd: Рабочая директория (каталог датасета / task.md).
        experiment_name: Имя эксперимента для версионирования кода.
        
    Returns:
        ExecutionResult с результатами выполнения, кодом возврата и временем.
    """
    base_dir = Path(__file__).resolve().parent
    workspace_dir = Path(cwd) if cwd else (base_dir / "workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{experiment_name}.py" if experiment_name else "agent_script.py"
    script_path = workspace_dir / filename

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_str)

    # Добавляем корень проекта в PYTHONPATH, чтобы сгенерированный код мог импортировать ml_core
    project_root = str(base_dir.parent)
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{current_pythonpath}" if current_pythonpath else project_root
    env["PYTHONIOENCODING"] = "utf-8"

    start_t = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace_dir),
            env=env,
        )
        duration = round(time.time() - start_t, 2)
        
        return ExecutionResult(
            success=(result.returncode == 0),
            stdout=result.stdout.strip()[-15000:],
            stderr=result.stderr.strip()[-15000:],
            returncode=result.returncode,
            duration=duration,
            script_path=str(script_path),
        )

    except subprocess.TimeoutExpired as e:
        duration = round(time.time() - start_t, 2)
        stdout_data = e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else (e.stdout or "")
        return ExecutionResult(
            success=False,
            stdout=stdout_data.strip()[-15000:],
            stderr=f"TimeoutExpired: Скрипт выполнялся дольше {timeout} секунд и был остановлен.",
            returncode=-1,
            duration=duration,
            script_path=str(script_path),
        )
    except Exception as e:
        duration = round(time.time() - start_t, 2)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"SystemError: Не удалось запустить процесс. Ошибка: {str(e)}",
            returncode=-2,
            duration=duration,
            script_path=str(script_path),
        )