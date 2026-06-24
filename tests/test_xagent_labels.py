import json

from xrouter_llm.xagent_labels import load_xagent_openrouter_labels


def test_load_xagent_labels_from_local_jsonl(tmp_path) -> None:
    path = tmp_path / "xagent.public.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "prompt_sha256": "abc123",
                    "prompt": "Current user request:\nhello",
                    "candidate_model": "google/gemini-2.5-flash-lite",
                    "candidate_usage": {"cost": 0.001},
                    "category": "skill",
                    "judge": {"score": 0.75, "can_complete": True},
                },
                {
                    "prompt_sha256": "abc123",
                    "prompt": "Current user request:\nhello",
                    "candidate_model": "deepseek/deepseek-v4-flash",
                    "candidate_usage": {"cost": 0.0005},
                    "category": "skill",
                    "judge": {"score": 0.25, "can_complete": False},
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = load_xagent_openrouter_labels(path)

    assert len(rows) == 2
    assert {row.prompt_id for row in rows} == {"xagent:abc123"}
    assert {row.model_id for row in rows} == {
        "deepseek/deepseek-v4-flash",
        "google/gemini-2.5-flash-lite",
    }
    assert [row.score for row in rows] == [0.75, 0.25]
    assert rows[0].cost_usd == 0.001
    assert rows[0].task == "xagent:skill"


def test_load_xagent_labels_rejects_unlabeled_candidates(tmp_path) -> None:
    path = tmp_path / "xagent.public.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt_sha256": "abc123",
                "prompt": "prompt",
                "category": "skill",
            }
        ),
        encoding="utf-8",
    )

    try:
        load_xagent_openrouter_labels(path)
    except ValueError as exc:
        assert "Missing judge.score" in str(exc)
    else:
        raise AssertionError("expected missing judge.score to fail")
