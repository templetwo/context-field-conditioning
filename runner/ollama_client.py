#!/usr/bin/env python3
"""
Inference client for CFC experiment.

Supports two backends:
  1. Ollama API (default) — no logprobs, uses alternative entropy estimation
  2. llama-server (llama.cpp) — native logprob support for Shannon entropy

The backend is selected via config.yaml or CLI flag.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "runner" / "config.yaml"


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    response_text: str
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    backend: str = ""

    @property
    def has_logprobs(self) -> bool:
        return len(self.logprobs) > 0


class OllamaClient:
    """Ollama API client for inference."""

    def __init__(self, model: str = None, base_url: str = None, config: dict = None):
        if config is None:
            config = self._load_config()

        self.model = model or config["model"]["name"]
        self.base_url = base_url or config["model"]["base_url"]
        self.temperature = config["inference"]["temperature"]
        self.num_predict = config["inference"]["num_predict"]
        self.top_p = config["inference"]["top_p"]
        self.repeat_penalty = config["inference"]["repeat_penalty"]

    @staticmethod
    def _load_config() -> dict:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def generate(self, prompt: str, system: str = None) -> InferenceResult:
        """Run a single inference and return the result."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
            },
            "stream": False,
        }
        if system:
            payload["system"] = system

        url = f"{self.base_url}/api/generate"
        start = time.monotonic()

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        result = InferenceResult(
            response_text=data.get("response", ""),
            total_tokens=data.get("eval_count", 0),
            latency_ms=elapsed_ms,
            model=self.model,
            backend="ollama",
        )

        # Attempt to extract logprobs (Ollama version dependent)
        for key in ("logprobs", "completion_probabilities", "tokens"):
            if key in data and data[key]:
                self._parse_logprobs(result, data[key])
                break

        return result

    def _parse_logprobs(self, result: InferenceResult, logprob_data):
        """Parse logprob data from Ollama response (format varies by version)."""
        if isinstance(logprob_data, list):
            for entry in logprob_data:
                if isinstance(entry, dict):
                    result.tokens.append(entry.get("token", ""))
                    result.logprobs.append(entry.get("logprob", 0.0))
                    if "top_logprobs" in entry:
                        result.top_logprobs.append(entry["top_logprobs"])
                elif isinstance(entry, (int, float)):
                    result.logprobs.append(float(entry))

    def health_check(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning("Model '%s' not found. Available: %s", self.model, models)
                return False
            return True
        except requests.RequestException:
            return False


class LlamaServerClient:
    """llama-server (llama.cpp) client with native logprob support."""

    def __init__(self, base_url: str = "http://localhost:8080", config: dict = None):
        if config is None:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f)

        self.base_url = base_url
        self.temperature = config["inference"]["temperature"]
        self.n_predict = config["inference"]["num_predict"]
        self.top_p = config["inference"]["top_p"]
        self.repeat_penalty = config["inference"]["repeat_penalty"]

    def generate(self, prompt: str, system: str = None) -> InferenceResult:
        """Run inference with logprob extraction."""
        full_prompt = prompt
        if system:
            full_prompt = f"[INST] {system}\n\n{prompt} [/INST]"

        payload = {
            "prompt": full_prompt,
            "temperature": self.temperature,
            "n_predict": self.n_predict,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "logprobs": True,
            "n_probs": 10,  # top-10 logprobs per token
        }

        url = f"{self.base_url}/completion"
        start = time.monotonic()

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("llama-server request failed: %s", e)
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()

        result = InferenceResult(
            response_text=data.get("content", ""),
            total_tokens=data.get("tokens_predicted", 0),
            latency_ms=elapsed_ms,
            model="ministral-14b",
            backend="llama-server",
        )

        # Parse completion_probabilities from llama.cpp
        probs = data.get("completion_probabilities", [])
        for token_data in probs:
            result.tokens.append(token_data.get("content", ""))
            # llama.cpp returns top_probs as list of {tok_str, prob}
            top = token_data.get("probs", [])
            if top:
                # The first entry is the selected token
                import math
                prob = top[0].get("prob", 1.0)
                result.logprobs.append(math.log(prob) if prob > 0 else float("-inf"))
                result.top_logprobs.append({t["tok_str"]: math.log(t["prob"]) for t in top if t.get("prob", 0) > 0})

        return result

    def health_check(self) -> bool:
        """Verify llama-server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False


def get_client(backend: str = "ollama", **kwargs):
    """Factory function to get the appropriate inference client."""
    if backend == "llama-server":
        return LlamaServerClient(**kwargs)
    return OllamaClient(**kwargs)
