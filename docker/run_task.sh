#!/usr/bin/env bash
# Р1. Запуск агента на одной задаче в условиях платформы: 8 CPU, 32 GB, лимит 3 часа.
# Пример: docker/run_task.sh "given_data/Регрессия"
# Ключи LLM берутся из окружения (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL).
set -euo pipefail
TASK_DIR="$(cd "$1" && pwd)"
docker build -q -f docker/Dockerfile -t automl-agent .
docker run --rm --cpus=8 --memory=32g \
  -e OPENAI_API_KEY -e OPENAI_BASE_URL -e OPENAI_MODEL -e AUTOML_FAST \
  -v "$TASK_DIR":/task \
  automl-agent timeout 3h python /app/solution.py /task/task.md
