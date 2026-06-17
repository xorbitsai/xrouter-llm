import pandas as pd

from xrouter_llm.routerbench import infer_routerbench_model_ids, routerbench_dataframe_to_rows


def test_routerbench_dataframe_to_rows_converts_wide_scores() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "p1",
                "prompt": "['system', 'question']",
                "eval_name": "toy",
                "model-a": 1.0,
                "model-b": 0.0,
                "model-a|model_response": "ok",
                "model-b|model_response": "bad",
                "model-a|total_cost": 0.1,
                "model-b|total_cost": 0.2,
                "oracle_model_to_route_to": "model-a",
            }
        ]
    )

    assert infer_routerbench_model_ids(dataframe) == ["model-a", "model-b"]

    rows = routerbench_dataframe_to_rows(dataframe)

    assert len(rows) == 2
    assert rows[0].prompt == "system\nquestion"
    assert rows[0].model_id == "model-a"
    assert rows[0].score == 1.0
    assert rows[0].cost_usd == 0.1
