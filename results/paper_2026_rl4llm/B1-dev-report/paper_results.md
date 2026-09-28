# Measured paper results

Only supplied, validated predictions are summarized. Unmeasured fields remain explicit.


## table1_main

| run_id | condition | training_seed | N_questions | N_rollouts | valid_finish_pct | correct_and_finished_pct | grounded_success_pct | mean_A | mean_G | unresolved | infrastructure_failures | correct_upper_bound_pct | grounded_upper_bound_pct | output_tokens_per_question | tool_calls_per_question | latency_s_per_question | cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1-dev-smoke | B1 | unmeasured | 12 | 12 | 16.6667 | 0 | 0 | 0.0000 | 0.0833 | 1 | 0 | 8.3333 | 8.3333 | 1936.1667 | 5.7500 | 13.1404 | unmeasured |


## table2_failures

| run_id | accepted_finish | plain_text | parse | validation | length | context | turn_limit | infrastructure | other | gold_page_read_pct | derived_calculator_pct | zero_citation_episodes | visible_citation_pct | median_turns | median_raw_tool_chars | p95_raw_tool_chars | median_visible_tool_chars | p95_visible_tool_chars | truncation_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1-dev-smoke | 2 | 0 | 0 | 0 | 4 | 0 | 6 | 0 | 0 | 75.0000 | unmeasured | 10 | 100.0000 | 8.0000 | 2471 | 9272 | 2522 | 3782 | 15.9420 |


## Limitations
Previously inspected evaluation split; small sample; document overlap must be reported from protocol.json; public-data pretraining contamination is unknown. Training seeds and question-level uncertainty measure different sources of variation. These outputs do not establish causality for observational tool-use correlations.
