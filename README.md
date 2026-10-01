# Qwen2.5-Coder-14B LoRA on MBPP

This project fine-tunes Qwen2.5-Coder-14B-Instruct with LoRA and evaluates executable Python correctness.

## Research question

> Does LoRA fine-tuning improve executable code correctness on a held-out MBPP test split?

## Critical dataset detail

The historical notebook loads the sanitized MBPP dataset and adds all 164 HumanEval examples to the training set.

Therefore HumanEval is **not held out** for this experiment. It may only be reported as a leakage diagnostic.

The MBPP test split is the appropriate primary evaluation set for the trained checkpoint.

## Historical training

The executed notebook reports 284 training examples and 257 evaluation examples, 5 epochs, effective batch size 8 and roughly 180 update steps.

The notebook's post-training evaluation consists of five hand-written prompts and does not run the model through the full MBPP executable test suite.

## Controlled evaluation

This branch adds an automated evaluator for the 257-example sanitized MBPP test split.

Primary metric:
- task pass rate under executable tests

Secondary metrics:
- challenge-test pass rate when available
- timeout/execution failures
- generation latency

Both base and adapter use deterministic decoding.

## Run

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_mbpp.py --mode base --limit 20 --output results/base_20.json
python scripts/evaluate_mbpp.py --mode adapter --adapter ./qwen-coder-final --limit 20 --output results/adapter_20.json
python scripts/analyze_results.py results/base_20.json results/adapter_20.json
```

For the full primary benchmark, remove `--limit`.

## Hypothesis

> LoRA fine-tuning improves MBPP executable task pass rate relative to the frozen base model.

This is falsifiable.

## Status

Implemented:
- LoRA training prototype
- leakage audit
- held-out MBPP executable evaluator
- base-vs-adapter comparison
- result analysis

Not yet demonstrated:
- full 257-task pass-rate improvement
- valid held-out HumanEval result