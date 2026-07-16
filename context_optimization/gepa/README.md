# IFBench-like GEPA prompt optimization

Start the Gym environment:

```bash
gym env start --config instruction_following_qwen.yaml
```

Run GEPA. It uses the same metric as the original GEPA/IFBench paper — the fractional
constraint pass rate — as both its search signal and the reported validation metric:

```bash
python optimize.py \
  --agent-url http://127.0.0.1:19754 \
  --train-data data/train.jsonl \
  --val-data data/validation.jsonl \
  --test-data data/test.jsonl \
  --grading-mode fraction \
  --max-calls 2000 \
  --reflection-model "$REFLECTION_MODEL" \
  --reflection-base-url "$REFLECTION_BASE_URL" \
  --reflection-api-key "$REFLECTION_API_KEY"
```

`fraction` is the IFBench/GEPA-paper metric: each rollout scores the fraction of its
constraints that pass, averaged over the set. Use `--grading-mode binary` to instead
select and report on the strict all-constraints response-level metric. The JSON curve
always uses the configured objective.

GEPA selects candidates on `--val-data` (its Pareto set), so those scores are optimistic.
The headline `baseline -> optimized` numbers (`baseline_test_acc` / `optimized_test_acc`
in the JSON) are therefore reported on the held-out `--test-data`; the val scores are kept
as secondary fields.

To collect ordinary Gym rollouts without optimization:

```bash
gym eval run --no-serve \
  --agent instruction_following_haystack_agent \
  --input data/test.jsonl \
  --output results/if_default_rollouts.jsonl \
  --num-repeats 1
```

This recipe exposes the external Gym agent as one synthetic DSPy predictor. It is a
valid single-stage GEPA setup, but it does not reproduce the GEPA paper's two-stage
IFBench answer-and-rewrite system or its out-of-distribution IFBench test set.
