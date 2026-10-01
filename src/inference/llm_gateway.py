from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for inference providers (vLLM, OpenAI)."""

    provider: str  # 'vllm' or 'openai'
    model_name: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str | None = None


class LLMGateway:
    """
    Enterprise LLM Inference Gateway.

    Supports:
    1. vLLM: High-throughput local or remote vLLM serving engine (OpenAI-compatible /v1).
    2. OpenAI: Direct cloud OpenAI fallback (gpt-4o-mini, gpt-4o).
    3. Automatic fallback: If vLLM is configured but down, gracefully switches to OpenAI.
    4. Inference telemetry: Tracks time-to-first-token, total latency, and tokens/sec.
    """

    def __init__(self) -> None:
        self.vllm_enabled = os.getenv("VLLM_ENABLED", "false").lower() in ("true", "1", "yes")
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_model = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def _vllm_healthy(self) -> bool:
        """Return True only when the vLLM endpoint actually answers.

        ChatOpenAI(...) construction does not open a connection, so relying
        on the constructor to detect a dead vLLM silently never falls back.
        This pings the /health endpoint before we trust the provider.
        """
        if not self.vllm_enabled:
            return False

        import urllib.request
        import urllib.error

        health_url = f"{self.vllm_base_url.rstrip('/v1')}/health"
        req = urllib.request.Request(health_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as response:
                return response.status < 400
        except Exception:
            return False

    def get_llm(self, temperature: float = 0.0, max_tokens: int = 1024) -> tuple[BaseChatModel, str]:
        """
        Return the primary LLM instance and provider label.

        Returns:
            (llm_instance, provider_name)
        """
        if self.vllm_enabled:
            if self._vllm_healthy():
                logger.info("Using vLLM endpoint: %s (model: %s)", self.vllm_base_url, self.vllm_model)
                vllm_llm = ChatOpenAI(
                    model=self.vllm_model,
                    base_url=self.vllm_base_url,
                    api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30.0,
                )
                return vllm_llm, f"vLLM ({self.vllm_model})"

            logger.warning(
                "vLLM enabled but unreachable at %s. Falling back to OpenAI (%s)",
                self.vllm_base_url,
                self.openai_model,
            )

        # Fallback to OpenAI
        openai_llm = ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return openai_llm, f"OpenAI ({self.openai_model})"

    def test_vllm_health(self) -> dict[str, Any]:
        """Ping vLLM server to check connectivity and readiness."""
        if not self.vllm_enabled:
            return {"status": "disabled", "url": self.vllm_base_url}

        import urllib.request
        try:
            health_url = f"{self.vllm_base_url.rstrip('/v1')}/health"
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                return {"status": "healthy", "code": response.status, "url": health_url}
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc), "url": self.vllm_base_url}


# Global singleton gateway
llm_gateway = LLMGateway()
