import json
import os
import socket
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.config import LLM_MODEL_NAME, OLLAMA_URL as DEFAULT_OLLAMA_URL


def _setting(name: str, default: str) -> str:
    environment_value = os.environ.get(name)
    if environment_value:
        return environment_value

    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


OLLAMA_URL = _setting(
    "OLLAMA_URL",
    DEFAULT_OLLAMA_URL,
)
MODEL_NAME = _setting("OLLAMA_MODEL", LLM_MODEL_NAME)
LLM_PROVIDER = _setting("LLM_PROVIDER", "ollama").lower()
GEMINI_API_KEY = _setting("GEMINI_API_KEY", "")
GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_RETRY_DELAYS = (1, 2, 4)


def _generate_with_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini is selected but GEMINI_API_KEY is not configured. Add it "
            "in Streamlit app settings → Secrets."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )
    result = None
    for attempt in range(len(GEMINI_RETRY_DELAYS) + 1):
        try:
            with urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            if error.code in {500, 502, 503, 504} and attempt < len(GEMINI_RETRY_DELAYS):
                time.sleep(GEMINI_RETRY_DELAYS[attempt])
                continue
            if error.code == 429:
                message = (
                    "Gemini rate limit or quota reached. Check the API project's "
                    "quota and billing, then retry."
                )
            elif error.code in {400, 403, 404}:
                message = (
                    f"Gemini returned HTTP {error.code}. Check that the API key is "
                    f"valid and model '{GEMINI_MODEL}' is available."
                )
            elif error.code in {500, 502, 503, 504}:
                message = (
                    f"Gemini is temporarily unavailable (HTTP {error.code}) after "
                    f"{attempt + 1} attempts. Please wait a moment and retry."
                )
            else:
                message = f"Gemini returned HTTP {error.code}."
            raise RuntimeError(message) from error
        except URLError as error:
            raise RuntimeError(
                "Could not connect to the Gemini API. Check the app host's internet "
                "connection and try again."
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                "Gemini took too long to respond. Please retry."
            ) from error
        except json.JSONDecodeError as error:
            raise RuntimeError("Gemini returned an unreadable response.") from error

    candidates = result.get("candidates", [])
    parts = (
        candidates[0].get("content", {}).get("parts", [])
        if candidates and isinstance(candidates[0], dict)
        else []
    )
    answer = "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()
    if not answer:
        raise RuntimeError(
            "Gemini returned no answer. The prompt may have been blocked; "
            "please try rephrasing the question."
        )
    return answer


def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured model provider."""
    if LLM_PROVIDER == "gemini":
        return _generate_with_gemini(prompt)
    if LLM_PROVIDER != "ollama":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'gemini'."
        )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}. Check that model "
            f"'{MODEL_NAME}' is installed and the endpoint is correct."
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. Make sure the Ollama "
            "server is running and reachable from the app. For a hosted app, "
            "localhost refers to the hosting server, not your computer; configure "
            "OLLAMA_URL to a reachable Ollama endpoint."
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(
            "Ollama took too long to respond. Check the server and model, then retry."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned an unreadable response. Please retry.") from error

    try:
        answer = result.get("response")
    except AttributeError as error:
        raise RuntimeError("Ollama returned an invalid response.") from error

    if not isinstance(answer, str):
        raise RuntimeError("Ollama returned a response without generated text.")
    answer = answer.strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer. Please retry.")
    return answer