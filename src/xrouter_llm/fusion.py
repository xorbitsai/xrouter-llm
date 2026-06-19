from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateAnswer:
    model_id: str
    answer: str


def build_fusion_prompt(
    prompt: str,
    answers: Mapping[str, str] | Iterable[CandidateAnswer],
) -> str:
    candidates = _coerce_answers(answers)
    if len(candidates) < 2:
        raise ValueError("Fusion requires at least two candidate answers")

    rendered_answers = "\n\n".join(
        f"[Model: {candidate.model_id}]\n{candidate.answer}"
        for candidate in candidates
    )

    return f"""You are a judge and synthesizer.

User request:
{prompt}

Candidate answers:

{rendered_answers}

Task:
1. Identify the strongest answer.
2. Merge non-conflicting strengths.
3. Fix obvious mistakes.
4. Return the final answer only."""


def _coerce_answers(
    answers: Mapping[str, str] | Iterable[CandidateAnswer],
) -> list[CandidateAnswer]:
    if isinstance(answers, Mapping):
        return [
            CandidateAnswer(model_id=str(model_id), answer=str(answer))
            for model_id, answer in answers.items()
        ]
    return [
        answer if isinstance(answer, CandidateAnswer) else CandidateAnswer(str(answer[0]), str(answer[1]))
        for answer in answers
    ]
