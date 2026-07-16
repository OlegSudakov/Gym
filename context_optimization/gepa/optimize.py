import argparse
import json
import sys

import dspy
import requests


def _answer_text(response):
    """Pull the final assistant text out of a NeMoGym Responses dict (mirrors the IF verifier)."""
    output = (response or {}).get("output") or []
    if not output:
        return ""
    content = (output[-1] or {}).get("content") or []
    if content and isinstance(content[0], dict):
        return content[0].get("text", "") or ""
    return ""


def _instruction_feedback(r, row, reward, answer):
    """Rich, instruction-aware feedback for GEPA: name the constraints the model violated/satisfied."""
    ids = r.get("instruction_id_list") or row.get("instruction_id_list") or []
    flags = r.get("follow_instruction_list") or []
    all_pass = bool(r.get("follow_all_instructions")) if "follow_all_instructions" in r else reward >= 1.0
    if not flags or len(flags) != len(ids):
        # No per-instruction breakdown available; fall back to a coarse signal.
        return all_pass, ("SUCCESS." if all_pass else f"FAILURE (reward={reward:.2f}).\nModel answer:\n{answer}")
    failed = [i for i, ok in zip(ids, flags) if not ok]
    passed = [i for i, ok in zip(ids, flags) if ok]
    if all_pass:
        return True, f"SUCCESS. All {len(ids)} constraints satisfied: {passed}"
    return False, (
        f"FAILURE (reward={reward:.2f}). The answer violated these constraints: {failed}. "
        f"Satisfied: {passed}.\n"
        f"Model answer:\n{answer}\n"
        "The answer must directly address the user's request while satisfying every violated "
        "constraint and keeping the satisfied ones."
    )


class GymAgent(dspy.Module):
    def __init__(self, agent_url, seed, timeout, grading_mode="binary"):
        super().__init__()
        self.agent_url = agent_url
        self.timeout = timeout
        self.grading_mode = grading_mode
        self.predict = dspy.Predict("task -> answer")
        self.predict.signature = self.predict.signature.with_instructions(seed)

    def forward(self, row):
        rcp = row.get("responses_create_params") or {}
        base_input = rcp.get("input") or [{"role": "user", "content": row.get("problem") or row.get("question") or ""}]
        task = base_input[-1]["content"]
        body = {
            **row,
            "grading_mode": self.grading_mode,
            "responses_create_params": {
                **rcp,
                "input": [{"role": "system", "content": self.predict.signature.instructions}] + base_input,
            },
        }
        try:
            r = requests.post(f"{self.agent_url}/run", json=body, timeout=self.timeout).json()
            reward = float(r.get("reward", 0.0))
            answer = _answer_text(r.get("response"))
            all_pass, feedback = _instruction_feedback(r, row, reward, answer)
        except requests.exceptions.ConnectionError:
            raise SystemExit(f"\nCannot reach agent at {self.agent_url}. Use the agent URL from your ng_run logs.")
        except Exception as e:
            reward, answer = 0.0, ""
            all_pass, feedback = False, f"agent call failed: {type(e).__name__}"
        if dspy.settings.trace is not None:
            dspy.settings.trace.append((self.predict, {"task": task}, dspy.Prediction(answer=answer)))
        # `success` (reward) drives GEPA's pareto selection; `all_pass` is the strict all-constraints metric.
        return dspy.Prediction(success=reward, all_pass=float(all_pass), trajectory=answer, feedback=feedback)


def metric_simple(example, pred, trace=None):
    # IFBench/GEPA-paper metric: fractional constraint pass rate, the same signal GEPA
    # optimizes on (metric_feedback). With --grading-mode binary this collapses to strict
    # all-constraints pass, matching pred.all_pass.
    return pred.success


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=pred.success, feedback=pred.feedback)


def load_dataset(path):
    rows = [json.loads(line) for line in open(path)]
    return [dspy.Example(row=r).with_inputs("row") for r in rows]


def run(args):
    lm = dspy.LM(
        f"openai/{args.reflection_model}",
        api_key=args.reflection_api_key,
        api_base=args.reflection_base_url,
        temperature=1.0,
        max_tokens=args.max_tokens,
        num_retries=10,
    )
    dspy.configure(lm=lm)

    try:
        requests.get(f"{args.agent_url}/", timeout=10).raise_for_status()
    except Exception:
        raise SystemExit(f"Agent not reachable at {args.agent_url}. Copy the agent URL from your ng_run logs.")

    train = load_dataset(args.train_data)
    val = load_dataset(args.val_data)
    test = load_dataset(args.test_data)
    print(f"optimizer={args.optimizer} train={len(train)} val={len(val)} test={len(test)}", flush=True)

    def build():
        return GymAgent(args.agent_url, args.seed, args.timeout, args.grading_mode)

    # GEPA selects candidates on `val` (its Pareto set), so val scores are optimistic. Report the
    # headline baseline/optimized numbers on held-out `test`; keep `val` numbers as secondary fields.
    evaluate = dspy.Evaluate(devset=val, metric=metric_simple, num_threads=args.num_threads, display_progress=True)
    evaluate_test = dspy.Evaluate(
        devset=test, metric=metric_simple, num_threads=args.num_threads, display_progress=True
    )
    seed_agent = build()
    baseline_val = evaluate(seed_agent).score / 100.0
    baseline = evaluate_test(seed_agent).score / 100.0

    if args.optimizer == "gepa":
        teleprompter = dspy.GEPA(
            metric=metric_feedback,
            max_metric_calls=args.max_calls,
            reflection_lm=lm,
            track_stats=True,
            num_threads=args.num_threads,
        )
        optimized = teleprompter.compile(build(), trainset=train, valset=val)
    else:
        from dspy.teleprompt import MIPROv2

        teleprompter = MIPROv2(metric=metric_simple, auto="light", prompt_model=lm, num_threads=args.num_threads)
        optimized = teleprompter.compile(
            build(),
            trainset=train,
            valset=val,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            requires_permission_to_run=False,
        )

    after_val = evaluate(optimized).score / 100.0
    after = evaluate_test(optimized).score / 100.0
    best_prompt = optimized.predict.signature.instructions
    print(f"\n{args.optimizer}: baseline {baseline} -> optimized {after} (held-out test)", flush=True)
    print(f"{args.optimizer}: baseline {baseline_val} -> optimized {after_val} (val, GEPA selection set)", flush=True)
    print("\n===== BEST PROMPT =====\n" + best_prompt)

    if args.out:
        result = {
            "optimizer": args.optimizer,
            "grading_mode": args.grading_mode,
            "seed_prompt": args.seed,
            "baseline_test_acc": baseline,
            "optimized_test_acc": after,
            "baseline_val_acc": baseline_val,
            "optimized_val_acc": after_val,
            "best_prompt": best_prompt,
            "train_n": len(train),
            "val_n": len(val),
            "test_n": len(test),
            "max_calls": args.max_calls,
            "candidates": candidates_from(optimized),  # [{candidate, prompt, val_acc, eval_calls, is_best}]
            "curve": curve_from(optimized),  # [{candidate, eval_calls, val_acc, topline}] for plotting
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote results + curve to {args.out}", flush=True)
    return optimized


def candidates_from(optimized):
    """Every candidate prompt GEPA proposed (not just the best), with its val score and the eval
    budget consumed at discovery. Empty for MIPRO / when track_stats is off."""
    dr = getattr(optimized, "detailed_results", None)
    programs = list(getattr(dr, "candidates", []) or [])
    if not programs:
        return []
    scores = list(getattr(dr, "val_aggregate_scores", []) or [])
    calls = list(getattr(dr, "discovery_eval_counts", []) or [])
    best_idx = getattr(dr, "best_idx", None)
    out = []
    for i, prog in enumerate(programs):
        out.append(
            {
                "candidate": i,
                "prompt": prog.predict.signature.instructions,
                "val_acc": scores[i] if i < len(scores) else None,
                "eval_calls": calls[i] if i < len(calls) else None,
                "is_best": i == best_idx,
            }
        )
    return out


def curve_from(optimized):
    """Per-candidate val-accuracy curve from DSPy GEPA's track_stats results, where `topline` is the running-max accuracy to plot against candidate index (empty for MIPRO)."""
    dr = getattr(optimized, "detailed_results", None)
    scores = list(getattr(dr, "val_aggregate_scores", []) or [])
    if not scores:
        return []
    calls = list(getattr(dr, "discovery_eval_counts", []) or [])
    curve, best = [], float("-inf")
    for i, s in enumerate(scores):
        best = max(best, s)
        curve.append(
            {"candidate": i, "eval_calls": calls[i] if i < len(calls) else None, "val_acc": s, "topline": best}
        )
    return curve


def main(default_optimizer=None):
    if default_optimizer and not any(a.startswith("--optimizer") for a in sys.argv):
        sys.argv += ["--optimizer", default_optimizer]
    p = argparse.ArgumentParser(description="DSPy-optimize a NeMo Gym agent's system prompt.")
    p.add_argument("--optimizer", choices=["gepa", "mipro"], default="gepa")
    p.add_argument("--agent-url", required=True, help="Gym agent /run endpoint, e.g. http://127.0.0.1:18123")
    p.add_argument(
        "--train-data",
        required=True,
        help="Gym-format JSONL trainset (responses_create_params.input + verifier fields)",
    )
    p.add_argument(
        "--val-data", required=True, help="Gym-format JSONL valset (responses_create_params.input + verifier fields)"
    )
    p.add_argument(
        "--test-data",
        default="data/test.jsonl",
        help="Held-out Gym-format JSONL testset for the headline baseline/optimized metric "
        "(GEPA selects candidates on --val-data, so val scores are optimistic)",
    )
    p.add_argument("--seed", default="Solve the problem.", help="Initial system prompt to optimize")
    p.add_argument(
        "--grading-mode",
        choices=["binary", "fraction"],
        default="fraction",
        help="Verifier reward used as both GEPA's search signal and the reported val metric "
        "(the IFBench/GEPA-paper metric): 'fraction' (per-constraint partial credit, matching the "
        "paper's fractional constraint pass rate) or 'binary' (strict all-or-nothing).",
    )
    p.add_argument("--reflection-model", required=True, help="Reflection/prompt LLM id")
    p.add_argument("--reflection-base-url", required=True, help="OpenAI-compatible base URL for the LLM")
    p.add_argument("--reflection-api-key", required=True)
    p.add_argument("--max-calls", type=int, default=1000, help="GEPA rollout budget")
    p.add_argument("--max-tokens", type=int, default=32000, help="Max tokens for the reflection/prompt LLM")
    p.add_argument("--num-threads", type=int, default=16, help="Concurrent rollouts (parallel agent/model queries)")
    p.add_argument("--timeout", type=int, default=60, help="Per-rollout timeout (s)")
    p.add_argument("--out", default="gepa_results.json", help="Write results + iteration curve here (JSON)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
