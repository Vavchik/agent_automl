"""Р4. Клиент LLM (Qwen на платформе) с бюджетом токенов.

Включается, если заданы OPENAI_API_KEY и OPENAI_MODEL (OPENAI_BASE_URL читает сам клиент openai).
Для локальной отладки можно указать Ollama: OPENAI_BASE_URL=http://localhost:11434/v1.
Если LLM недоступна, кончился бюджет или было 3 ошибки подряд — ask() возвращает None,
и агент работает на эвристиках. Падать из-за LLM пайплайн не должен.
"""
from __future__ import annotations

import json
import os
import re

import config
from runtime.logger import log


class LLM:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL")
        self.enabled = bool(os.getenv("OPENAI_API_KEY") and self.model)
        self.used_tokens = 0
        self.failures = 0
        self._client = None

    def available(self) -> bool:
        return self.enabled and self.used_tokens < config.LLM_TOKEN_BUDGET and self.failures < 3

    def ask(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.3) -> str | None:
        if not self.available():
            return None
        try:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(timeout=config.LLM_TIMEOUT_SEC, max_retries=1)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            self.failures += 1
            log.warning(f"LLM недоступна ({self.failures}/3): {e}")
            return None
        self.failures = 0
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        self.used_tokens += usage.total_tokens if usage else (len(system) + len(user) + len(text)) // 3
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def ask_json(self, system: str, user: str, max_tokens: int = 1000) -> dict | None:
        text = self.ask(system, user, max_tokens=max_tokens, temperature=0.1)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.S)
        try:
            return json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return None


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    code = match.group(1) if match else text
    return code if "def transform_features" in code else None


llm = LLM()
