"""DeepEval adapter for the local Ollama chat API."""

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_CONNECT_TIMEOUT = 5
OLLAMA_REQUEST_TIMEOUT = 180


class OllamaJudge(DeepEvalBaseLLM):
    """Use Ollama as a DeepEval judge model."""

    def __init__(self, model_name: str = OLLAMA_MODEL) -> None:
        self.host = OLLAMA_HOST
        self._check_server(model_name)
        super().__init__(model_name=model_name)

    def _check_server(self, model_name: str) -> None:
        request = Request(f"{self.host}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=OLLAMA_CONNECT_TIMEOUT) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama health check failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Ollama is unavailable at {self.host}; start the Ollama service."
            ) from error

        models = result.get("models") if isinstance(result, dict) else None
        if not isinstance(models, list):
            raise ValueError("Ollama /api/tags returned an invalid model list.")
        installed_names = {
            item.get("name")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if model_name not in installed_names:
            raise RuntimeError(
                f"Ollama model '{model_name}' is not installed. "
                f"Install it with `ollama pull {model_name}`."
            )

    def load_model(self) -> "OllamaJudge":
        return self

    def generate(self, prompt: str, schema: Any = None) -> Any:
        message = f"/no_think\n{prompt}"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 1024},
        }
        if schema is not None:
            json_schema = schema.model_json_schema()
            message += (
                "\n\nReturn only a JSON object matching this schema:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            payload["messages"] = [{"role": "user", "content": message}]
            payload["format"] = json_schema

        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Ollama request failed at {self.host} or exceeded "
                f"{OLLAMA_REQUEST_TIMEOUT} seconds."
            ) from error

        response_message = result.get("message") if isinstance(result, dict) else None
        content = (
            response_message.get("content")
            if isinstance(response_message, dict)
            else None
        )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned an empty or invalid response.")
        if schema is None:
            return content
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        return schema(**json.loads(cleaned))

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name
