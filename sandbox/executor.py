import os
import subprocess
import sys
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str

def execute_code(code_str: str, timeout: int = 60) -> ExecutionResult:
    """
    Безопасно сохраняет и запускает переданный Python-код в отдельном процессе.
    
    Args:
        code_str: Строка с кодом, который написал LLM-агент.
        timeout: Максимальное время выполнения в секундах.
        
    Returns:
        ExecutionResult с результатами выполнения и ошибками (если есть).
    """
    # 1. Формируем абсолютные пути, чтобы не зависеть от того, 
    # откуда вызывается main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.join(base_dir, "workspace")
    script_path = os.path.join(workspace_dir, "agent_script.py")

    # Создаем папку workspace, если её еще нет
    os.makedirs(workspace_dir, exist_ok=True)

    # 2. Сохраняем код в файл
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_str)

    # 3. Запускаем код в изолированном процессе
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,       # Возвращает строки вместо байтов
            timeout=timeout
        )
        
        # 4. Обработка результатов
        if result.returncode == 0:
            return ExecutionResult(
                success=True,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip()
            )
        else:
            return ExecutionResult(
                success=False,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip() # Здесь будет лежать Traceback ошибки
            )

    except subprocess.TimeoutExpired as e:
        # Защита от бесконечных циклов (while True) или тяжелых вычислений
        stdout_data = e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else (e.stdout or "")
        return ExecutionResult(
            success=False,
            stdout=stdout_data.strip(),
            stderr=f"TimeoutExpired: Скрипт выполнялся дольше {timeout} секунд и был остановлен."
        )
    except Exception as e:
        # Защита от системных ошибок (например, не найден интерпретатор)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"SystemError: Не удалось запустить процесс. Ошибка: {str(e)}"
        )