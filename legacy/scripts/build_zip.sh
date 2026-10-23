#!/usr/bin/env bash
# Р1. Собирает solution.zip строго под регламент Boosters:
# Все файлы в корне архива, без папок (solution.py + requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/bundle_solution.py
unzip -l solution.zip
echo "Готово: solution.zip собран и проверен для отправки на платформу."
