"""OpenRouter client. Every LLM call (Claude agents and Gemini listening)
goes through here, so spend is tracked in one place."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import requests


class LLMError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class Spend:
    def __init__(self, cap_usd: float, log_path: Path | None = None):
        self.cap = cap_usd
        self.total = 0.0
        self.calls = 0
        self.log_path = log_path

    def add(self, usd: float, entry: dict):
        self.total += usd
        self.calls += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "usd": usd, "total": self.total, **entry}) + "\n")

    def check(self):
        if self.total >= self.cap:
            raise BudgetExceeded(f"LLM spend ${self.total:.2f} reached cap ${self.cap:.2f}")


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating code fences / prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i: j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON in reply: {text[:300]!r}")


def audio_part(data: bytes, fmt: str) -> dict:
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(data).decode("ascii"), "format": fmt}}


class OpenRouter:
    def __init__(self, cfg: dict, spend: Spend):
        m = cfg["models"]
        self.base = m["base_url"].rstrip("/")
        self.key = os.environ.get(m["api_key_env"], "")
        self.timeout = m["timeout_s"]
        self.retries = m["max_retries"]
        self.spend = spend

    @property
    def available(self) -> bool:
        return bool(self.key)

    def chat_tools(self, model: str, messages: list, tools: list, tool_choice="auto",
                   max_tokens: int = 8000, tag: str = "", temperature: float | None = None) -> dict:
        """One tool-calling turn. Returns the assistant message dict (content and/or tool_calls,
        plus any reasoning_details, which must be sent back unchanged on the next turn)."""
        if not self.key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.spend.check()
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                "max_tokens": max_tokens, "usage": {"include": True}}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "mixloop"}
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=headers,
                                  json=body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(2 ** attempt * 3)
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                d = r.json()
                if "error" in d:
                    last = f"API error: {d['error']}"
                    time.sleep(2 ** attempt * 3)
                    continue
                msg = d["choices"][0]["message"]
                usage = d.get("usage", {}) or {}
                self.spend.add(float(usage.get("cost") or 0.0),
                               {"model": model, "tag": tag,
                                "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")})
                return msg
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt * 3)
        raise LLMError(f"{tag or model}: failed after retries: {last}")

    def chat(self, model: str, messages: list, schema: dict | None = None,
             max_tokens: int = 8000, temperature: float | None = None,
             tag: str = "", reasoning: str | None = None) -> tuple[object, dict]:
        """Returns (parsed JSON if schema else text, raw message dict)."""
        if not self.key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.spend.check()
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "usage": {"include": True}}
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning:
            body["reasoning"] = {"effort": reasoning}
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "strict": False,
                                                       "schema": schema}}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "mixloop"}
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=headers,
                                  json=body, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(2 ** attempt * 3)
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                d = r.json()
                if "error" in d:
                    last = f"API error: {d['error']}"
                    time.sleep(2 ** attempt * 3)
                    continue
                msg = d["choices"][0]["message"]
                usage = d.get("usage", {}) or {}
                self.spend.add(float(usage.get("cost") or 0.0),
                               {"model": model, "tag": tag,
                                "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens")})
                text = msg.get("content") or ""
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                if schema is None:
                    return text, msg
                try:
                    return extract_json(text), msg
                except LLMError as e:
                    last = str(e)
                    messages = messages + [{"role": "assistant", "content": text},
                                           {"role": "user", "content": "Reply with valid JSON only."}]
                    body["messages"] = messages
                    continue
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt * 3)
        raise LLMError(f"{tag or model}: failed after retries: {last}")
