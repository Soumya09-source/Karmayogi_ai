"""
Thin client for a locally-running Ollama instance.

Assumes `ollama serve` is already running (default: http://localhost:11434)
and the target model has been pulled, e.g.:
    ollama pull llama3.1
"""

from __future__ import annotations

import json
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_0"
REQUEST_TIMEOUT_SECONDS = 120


def generate_json(prompt: str, model: str = DEFAULT_MODEL, schema: dict | None = None) -> dict:
    """
    Calls Ollama's /api/generate with structured output.

    If `schema` is given, it's passed as the `format` value directly (a
    real JSON Schema, not just the string "json") — this constrains
    generation at the token/grammar level, not just via prompt wording.
    Critically, an array schema with "minItems"/"maxItems" set actually
    forces the model to fill that many array entries structurally, which
    plain prompt instructions ("generate exactly N") were not reliably
    achieving with llama3.1 in testing.

    If `schema` is omitted, falls back to format="json" (valid JSON,
    shape not constrained) — used for the self-consistency check, where
    we only want a short free-text-ish answer, not a structured object.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": schema if schema is not None else "json",
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    raw_text = response.json()["response"]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Ollama did not return valid JSON despite a format constraint. "
            f"Raw output: {raw_text[:500]}"
        ) from e


def mcq_array_schema(count: int) -> dict:
    """
    JSON Schema for an array of exactly `count` MCQ objects. Passed as
    `format` to generate_json() to structurally force the array length,
    rather than relying on the model choosing to comply with a prompt
    instruction.
    """
    mcq_item_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                },
            },
            "correct_option_id": {"type": "string"},
            "explanation": {"type": "string"},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        },
        "required": ["question", "options", "correct_option_id", "explanation", "difficulty"],
    }
    return {
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": mcq_item_schema,
    }


def generate_text(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Plain text generation, no JSON constraint — used for the
    self-consistency re-derivation step where we want a short free-text
    answer to compare, not a structured object."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
