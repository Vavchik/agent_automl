"""
Модуль взаимодействия с языковыми моделями (LLM).

Предоставляет:
- `ask_llm`: выполнение запроса к OpenAI/Ollama с настраиваемой температурой и моделью;
- `extract_code`: надежное извлечение блоков Python-кода из markdown;
- `extract_json`: надежное извлечение и валидация JSON из ответа модели.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from openai import OpenAI

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")

_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    """Возвращает инициализированный синглтон-клиент OpenAI."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
        )
    return _client


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """
    Отправляет системный и пользовательский промпты в LLM и возвращает ответ.
    """
    client = get_llm_client()
    target_model = model or DEFAULT_MODEL

    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


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
