"""
Тесты для модуля песочницы и исполнения скриптов (sandbox.executor).
"""

from pathlib import Path
from sandbox.executor import execute_code


def test_executor_simple_success(tmp_path: Path):
    code = "print('Hello from Sandbox!')"
    result = execute_code(code, cwd=str(tmp_path), experiment_name="test_success")
    assert result.success is True
    assert result.returncode == 0
    assert "Hello from Sandbox!" in result.stdout
    assert Path(result.script_path).exists()


def test_executor_error_capture(tmp_path: Path):
    code = "raise ValueError('Test intentional failure')"
    result = execute_code(code, cwd=str(tmp_path), experiment_name="test_fail")
    assert result.success is False
    assert result.returncode != 0
    assert "ValueError: Test intentional failure" in result.stderr


def test_executor_timeout(tmp_path: Path):
    code = "import time; time.sleep(5)"
    result = execute_code(code, timeout=1, cwd=str(tmp_path), experiment_name="test_timeout")
    assert result.success is False
    assert "TimeoutExpired" in result.stderr
