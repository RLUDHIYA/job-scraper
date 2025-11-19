import time
import random
import re
import logging
from typing import Optional, Any

from google import genai
from google.genai import types

import config


_clients: list[genai.Client] = []


def _init_clients() -> None:
    if _clients:
        return
    keys = [
        config.GEMINI_FIRST_API_KEY,
        config.GEMINI_SECOND_API_KEY,
        getattr(config, "GEMINI_THIRD_API_KEY", None),
        getattr(config, "GEMINI_FOURTH_API_KEY", None),
    ]
    for k in keys:
        if k:
            _clients.append(genai.Client(api_key=k))
    if not _clients:
        logging.warning("No Gemini API keys configured; client wrapper will not function")


def get_client(index: int = 0) -> genai.Client:
    _init_clients()
    if not _clients:
        raise RuntimeError("No Gemini clients available")
    return _clients[index % len(_clients)]


def _extract_retry_seconds(message: str) -> Optional[float]:
    m = re.search(r"retryDelay\W*(\d+)s", message)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"Please retry in\s*([0-9.]+)s", message)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return None


def generate_content_resilient(
    contents: Any,
    *,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    response_mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
    temperature: Optional[float] = None,
    max_attempts: int = 6,
    client_index: int = 0,
) -> Any:
    client = get_client(client_index)
    attempt = 0
    backoff_base = 2.0
    backoff_max = 60.0
    while attempt < max_attempts:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model or config.GEMINI_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                    response_mime_type=response_mime_type,
                    response_schema=response_schema,
                ),
            )
            return response
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                retry_seconds = _extract_retry_seconds(msg) or float(config.GEMINI_REQUEST_DELAY_SECONDS)
                logging.warning(f"Gemini rate limit hit; retrying in {retry_seconds:.2f}s (attempt {attempt}/{max_attempts})")
                time.sleep(retry_seconds + random.uniform(0, 0.5))
                continue
            if "UNAVAILABLE" in msg or "503" in msg or "overloaded" in msg:
                delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max) + random.uniform(0, 1.0)
                logging.warning(f"Gemini unavailable; backoff {delay:.2f}s (attempt {attempt}/{max_attempts})")
                time.sleep(delay)
                continue
            logging.error(f"Gemini error without retry: {msg}")
            raise
    raise RuntimeError("Exceeded max attempts calling Gemini")