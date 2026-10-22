# Р1. Окружение как на платформе: Python 3.12 + их requirements.txt.
# Сборка:  docker build -f docker/Dockerfile -t automl-agent .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt pytest
COPY . /app
