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
        self.model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1")
        self.backup_url = "https://api.alfahack-gateway26.ru/v1"
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.used_tokens = 0
        self.failures = 0

    def available(self) -> bool:
        return self.enabled and self.used_tokens < config.LLM_TOKEN_BUDGET and self.failures < 3

    def _get_client(self, url: str):
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=url, timeout=config.LLM_TIMEOUT_SEC, max_retries=1)

    def ask(self, system: str, user: str, max_tokens: int = 3500, temperature: float = 0.3) -> str | None:
        if not self.available():
            return None
        endpoints = [self.base_url]
        if self.base_url != self.backup_url:
            endpoints.append(self.backup_url)
        for ep in endpoints:
            try:
                client = self._get_client(ep)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.failures = 0
                msg = response.choices[0].message
                text = msg.content or ""
                if not text and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    text = msg.reasoning_content
                usage = getattr(response, "usage", None)
                self.used_tokens += usage.total_tokens if usage else (len(system) + len(user) + len(text)) // 3
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            except Exception as e:
                log.warning(f"Ошибка LLM ({ep}): {e}")
        self.failures += 1
        return None

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
    xml_match = re.search(r"<python_code>\s*(.*?)\s*</python_code>", text, flags=re.S)
    if xml_match:
        code = xml_match.group(1).strip()
    else:
        code = text.strip()
    md_match = re.search(r"```(?:python)?\s*(.*?)\s*```", code, flags=re.S)
    if md_match:
        code = md_match.group(1).strip()
    idx = code.find("def transform_features")
    if idx >= 0:
        return code[idx:].strip()
    return None


def extract_hypothesis(text: str | None) -> str:
    if not text:
        return "Признаки от LLM"
    xml_match = re.search(r"<hypothesis>\s*(.*?)\s*</hypothesis>", text, flags=re.S)
    if xml_match:
        return xml_match.group(1).strip().replace("\n", " ")[:300]
    before_code = text.split("```")[0].split("<python_code>")[0].strip()
    clean = re.sub(r"<thinking>.*?</thinking>", "", before_code, flags=re.S).strip()
    return clean.replace("\n", " ")[:300] or "Признаки от LLM"


def extract_thinking(text: str | None) -> str:
    if not text:
        return ""
    xml_match = re.search(r"<thinking>\s*(.*?)\s*</thinking>", text, flags=re.S)
    return xml_match.group(1).strip() if xml_match else ""


llm = LLM()
