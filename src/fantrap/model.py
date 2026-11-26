"""Model interface for SQL generation: a scripted offline model and an OpenAI-compatible adapter."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol

import httpx
import yaml


class ModelError(Exception):
    """The model could not produce SQL."""


class SqlModel(Protocol):
    def generate(self, question: str, schema_prompt: str, role: str, feedback: str | None) -> str:
        """Return one SQL statement for `question`. `feedback` carries repair hints, else None."""
        ...


class ScriptedModel:
    """Deterministic model that replays recorded attempts.

    `responses` maps a question id to its attempts in order. Once the attempts run past the list,
    the last one is returned again. A call with `feedback=None` starts a new run for that question.
    """

    def __init__(self, responses: dict[str, list[str]], question_ids: dict[str, str]) -> None:
        self._responses = responses
        self._question_ids = question_ids
        self._attempt: dict[str, int] = {}

    @classmethod
    def from_files(cls, responses_path: str | Path, questions_path: str | Path) -> ScriptedModel:
        with open(responses_path, encoding="utf-8") as handle:
            responses = {str(k): [str(s) for s in v] for k, v in yaml.safe_load(handle).items()}
        with open(questions_path, encoding="utf-8") as handle:
            questions = yaml.safe_load(handle)["questions"]
        return cls(responses, {q["question"]: q["id"] for q in questions})

    def _resolve(self, question: str) -> str:
        if question in self._question_ids:
            return self._question_ids[question]
        if question in self._responses:
            return question
        raise ModelError(f"no scripted response for question: {question!r}")

    def generate(self, question: str, schema_prompt: str, role: str, feedback: str | None) -> str:
        qid = self._resolve(question)
        attempts = self._responses.get(qid)
        if not attempts:
            raise ModelError(f"no scripted attempts recorded for {qid}")
        index = 0 if feedback is None else self._attempt.get(qid, 0) + 1
        self._attempt[qid] = index
        return attempts[min(index, len(attempts) - 1)]


_SYSTEM = (
    "You write one DuckDB SQL SELECT statement that answers the user's question. "
    "Reply with the SQL only. Never write data, never read files, and only use the tables "
    "listed below. When a query joins a table to its child rows (one-to-many) and sums or counts "
    "the parent's column, the total is inflated: aggregate the child table in a subquery "
    "grouped by the join key first.\n\n"
)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply that may wrap it in a markdown fence."""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


class OpenAICompatModel:
    """Chat-completions client for any OpenAI-compatible endpoint, configured by environment."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, client: httpx.Client | None = None) -> OpenAICompatModel:
        base = os.environ.get("FANTRAP_LLM_BASE_URL")
        key = os.environ.get("FANTRAP_LLM_API_KEY")
        model = os.environ.get("FANTRAP_LLM_MODEL")
        missing = [
            name
            for name, value in (
                ("FANTRAP_LLM_BASE_URL", base),
                ("FANTRAP_LLM_API_KEY", key),
                ("FANTRAP_LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ModelError("set " + ", ".join(missing) + " to use the OpenAI-compatible model")
        return cls(str(base), str(key), str(model), client=client)

    def generate(self, question: str, schema_prompt: str, role: str, feedback: str | None) -> str:
        user = f"Role: {role}\nQuestion: {question}"
        if feedback:
            user += f"\n\n{feedback}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM + schema_prompt},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ModelError(f"model request failed: {exc}") from exc
        return extract_sql(content)
