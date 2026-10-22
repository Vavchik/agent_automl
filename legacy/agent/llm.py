"""
Модуль взаимодействия с языковыми моделями (LLM).

Поддерживает:
1. Локальный Ollama (http://localhost:11434) через прямой HTTP REST API или OpenAI-совместимый интерфейс.
2. Облачный / Платформенный OpenAI-совместимый API (Qwen3.8-27B на Alfa Hack).
3. `ask_llm`: выполнение запроса к модели с настраиваемой температурой, повторными попытками и логгированием.
4. `extract_code`: надежное извлечение блоков Python-кода из markdown.
5. `extract_json`: надежное извлечение и валидация JSON из ответа модели.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional
import urllib.request
import urllib.error

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger("agent.llm")

# Конфигурация Ollama / OpenAI
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", f"{OLLAMA_HOST}/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", DEFAULT_OLLAMA_MODEL))

_client: Optional[OpenAI] = None

# Трекинг расхода токенов для соблюдения лимитов
token_usage_tracker = {
    "total_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def get_llm_client() -> Optional[Any]:
    """Возвращает OpenAI-клиент или None, если библиотека не установлена."""
    global _client
    if OpenAI is None:
        return None
    if _client is None:
        try:
            _client = OpenAI(
                base_url=DEFAULT_BASE_URL,
                api_key=DEFAULT_API_KEY,
                timeout=120.0,
                max_retries=2,
            )
        except Exception as e:
            logger.warning(f"Не удалось инициализировать OpenAI клиент: {e}")
            _client = None
    return _client


def _query_ollama_native(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.1,
    timeout_sec: float = 120.0,
) -> str:
    """
    Прямой запрос к нативному API Ollama (/api/chat) через urllib (без сторонних зависимостей).
    """
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    logger.info(f"[Ollama] Отправка запроса к модели '{model}' ({len(user_prompt)} симв.)...")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        content = res_data.get("message", {}).get("content", "")

        # Обновляем счетчик токенов при наличии
        prompt_eval_count = res_data.get("prompt_eval_count", 0)
        eval_count = res_data.get("eval_count", 0)
        token_usage_tracker["total_calls"] += 1
        token_usage_tracker["prompt_tokens"] += prompt_eval_count
        token_usage_tracker["completion_tokens"] += eval_count
        token_usage_tracker["total_tokens"] += prompt_eval_count + eval_count

        logger.info(
            f"[Ollama] Ответ получен за {elapsed:.2f}с ({len(content)} симв., "
            f"eval_tokens={eval_count})"
        )
        return content


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = 4096,
) -> str:
    """
    Универсальная функция запроса к LLM:
    1. Если заданы облачные ключи платформы (OPENAI_API_KEY != 'dummy-key' и != 'ollama'), вызывает OpenAI API.
    2. По умолчанию работает через локальный Ollama (нативно или через OpenAI-совместимый эндпоинт).
    3. При сетевых сбоях логирует детальные ошибки.
    """
    target_model = model or DEFAULT_MODEL
    has_platform_key = (
        bool(DEFAULT_API_KEY)
        and DEFAULT_API_KEY not in ("dummy-key", "ollama")
        and "localhost" not in DEFAULT_BASE_URL
        and "127.0.0.1" not in DEFAULT_BASE_URL
    )

    # 1. Попытка через платформенный OpenAI-совместимый API (если задан реальный ключ)
    if has_platform_key:
        client = get_llm_client()
        if client is not None:
            try:
                t0 = time.time()
                logger.info(f"[OpenAI-Proxy] Запрос к {target_model}...")
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response.usage:
                    token_usage_tracker["total_calls"] += 1
                    token_usage_tracker["prompt_tokens"] += response.usage.prompt_tokens or 0
                    token_usage_tracker["completion_tokens"] += response.usage.completion_tokens or 0
                    token_usage_tracker["total_tokens"] += response.usage.total_tokens or 0
                logger.info(f"[OpenAI-Proxy] Ответ получен за {time.time() - t0:.2f}с")
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning(f"Ошибка вызова OpenAI Proxy: {exc}. Переключение на локальный Ollama...")

    # 2. Локальный Ollama (нативный API)
    try:
        return _query_ollama_native(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=target_model,
            temperature=temperature,
        )
    except Exception as ollama_err:
        logger.warning(f"[Ollama] Не удалось получить ответ от нативного API ({ollama_err}). Пробуем OpenAI endpoint...")

    # 3. Fallback: Ollama через OpenAI клиент
    client = get_llm_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Все способы обращения к LLM ({target_model}) завершились сбоем: {e}")
            raise

    raise RuntimeError(
        f"Не удалось связаться с LLM через Ollama ({OLLAMA_HOST}) или OpenAI. "
        "Убедитесь, что сервис Ollama запущен (`ollama serve`)."
    )


def extract_code(llm_response: str) -> str:
    """
    Вырезает блок Python-кода из markdown ответа (между ```python и ```).
    Если разметки нет, возвращает текст целиком.
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return llm_response.strip()


def extract_json(llm_response: str) -> dict[str, Any]:
    """
    Извлекает JSON-объект из ответа языковой модели.
    Корректно обрабатывает блоки ```json ... ```, чистый текст и некритичные опечатки.
    """
    cleaned = llm_response.strip()

    # Поиск блока ```json ... ``` или ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # Поиск от первой { до последней }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start : end + 1]
        else:
            json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as err:
        # Попытка подчистить запятые перед закрывающей скобкой
        fixed_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            return json.loads(fixed_str)
        except Exception:
            raise ValueError(f"Не удалось распарсить JSON из ответа LLM: {err}\nТекст ответа:\n{llm_response}") from err
