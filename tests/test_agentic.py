import json

from xrouter_llm.agentic import load_agent_psychometrics


def _write(base, dataset, responses, tasks=None):
    d = base / "data" / dataset
    d.mkdir(parents=True)
    with (d / "responses.jsonl").open("w") as f:
        for subject_id, resp in responses.items():
            # responses is stored as a python-dict repr string in the source repo
            f.write(json.dumps({"subject_id": subject_id, "responses": repr(resp)}) + "\n")
    if tasks is not None:
        with (d / "tasks.jsonl").open("w") as f:
            for tid, text in tasks.items():
                f.write(json.dumps({"task_id": tid, "problem_statement": text}) + "\n")


def test_load_with_local_tasks(tmp_path):
    _write(
        tmp_path,
        "terminalbench",
        responses={"agentA": {"t1": 1, "t2": 0}, "agentB": {"t1": 0, "t2": 0}},
        tasks={"t1": "implement a sampler", "t2": "fix the build"},
    )
    rows = load_agent_psychometrics(tmp_path, "terminalbench")
    assert len(rows) == 4
    assert {r.model_id for r in rows} == {"agentA", "agentB"}
    assert {r.prompt for r in rows} == {"implement a sampler", "fix the build"}
    assert all(r.task == "terminalbench" for r in rows)
    t1 = [r.score for r in rows if r.prompt == "implement a sampler"]
    assert sorted(t1) == [0.0, 1.0]


def test_external_task_text_and_skip_missing(tmp_path):
    _write(
        tmp_path,
        "swebench_verified",
        responses={"sub1": {"inst-1": 1, "inst-missing": 1}},
    )
    rows = load_agent_psychometrics(
        tmp_path, "swebench_verified", task_text={"inst-1": "a problem statement"}
    )
    # inst-missing has no text -> skipped
    assert len(rows) == 1
    assert rows[0].prompt == "a problem statement"
    assert rows[0].prompt_id == "agentpsych:swebench_verified:inst-1"
