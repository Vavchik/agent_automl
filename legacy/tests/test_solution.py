"""
Тесты точки входа solution.py.
"""

from pathlib import Path
import subprocess
import sys
import pytest


def test_solution_cli_help():
    """Проверяет, что python solution.py --help отрабатывает без ошибок."""
    res = subprocess.run(
        [sys.executable, "solution.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Alfa Hack AutoML Agent Solution" in res.stdout
