"""Р4. Клиент LLM (Qwen на платформе) с бюджетом токенов.

Ключ: переменная OPENAI_API_KEY. На платформе её (вместе с OPENAI_BASE_URL и OPENAI_MODEL)
выставляет сама проверяющая система, как для baseline.py. Ключ в код НЕ вписываем.
Если ключа нет, LLM выключена и агент работает на эвристиках — падать из-за LLM пайплайн не должен.
Если основной адрес не отвечает, пробуем резервный (config.LLM_BACKUP_URL).
"""
from __future__ import annotations

import json
import os
import re

import config
from runtime.logger import log


class LLM:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL") or config.LLM_MODEL_DEFAULT
        self.base_url = os.getenv("OPENAI_BASE_URL") or config.LLM_BASE_URL_DEFAULT
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.used_tokens = 0
        self.calls = 0
        self.failures = 0
        self._clients: dict[str, object] = {}

    def describe(self) -> str:
        if not self.enabled:
            return "LLM выключена: нет OPENAI_API_KEY — работаем на эвристиках"
        return f"LLM включена: модель {self.model}, адрес {self.base_url}"

    def available(self) -> bool:
        return self.enabled and self.used_tokens < config.LLM_TOKEN_BUDGET and self.failures < 3

    def _client(self, url: str):
        if url not in self._clients:
            from openai import OpenAI
            self._clients[url] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=url,
                                        timeout=config.LLM_TIMEOUT_SEC, max_retries=1)
        return self._clients[url]

    def ask(self, system: str, user: str, max_tokens: int = 2500, temperature: float = 0.4) -> str | None:
        if not self.available():
            return None
        if config.LLM_NO_THINK:
            user += "\n/no_think"
        urls = [self.base_url] + ([config.LLM_BACKUP_URL] if config.LLM_BACKUP_URL != self.base_url else [])
        for url in urls:
            try:
                response = self._client(url).chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                log.warning(f"LLM ({url}) не ответила: {e}")
                continue
            self.failures = 0
            self.calls += 1
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            self.used_tokens += usage.total_tokens if usage else (len(system) + len(user) + len(text)) // 3
            if response.choices[0].finish_reason == "length":
                log.warning("Ответ LLM обрезан по max_tokens")
            return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()
        self.failures += 1
        log.warning(f"LLM недоступна ({self.failures}/3)")
        return None

    def ask_json(self, system: str, user: str, max_tokens: int = 800) -> dict | None:
        text = self.ask(system, user, max_tokens=max_tokens, temperature=0.1)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.S)
        try:
            return json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return None


def extract_tag(text: str | None, tag: str) -> str | None:
    if not text:
        return None
    match = re.search(rf"<{tag}>(.*?)(</{tag}>|$)", text, flags=re.S)
    return match.group(1).strip() if match else None


def extract_code(text: str | None) -> str | None:
    """Код из <code>...</code>, ```python ...``` или весь ответ, если в нём есть transform_features."""
    if not text:
        return None
    code = extract_tag(text, "code")
    if code is None:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
        code = match.group(1) if match else text
    code = re.sub(r"^```(?:python)?|```$", "", code.strip()).strip()
    if "```" in code:
        code = re.sub(r"```(?:python)?\s*", "", code)
        code = re.sub(r"```", "", code)
    if "def transform_features" in code:
        idx = code.find("def transform_features")
        return code[idx:].strip()
    return None


llm = LLM()
