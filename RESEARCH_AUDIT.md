# Research Audit — Qwen2.5-Coder 14B LoRA

## Research question

> Does LoRA fine-tuning improve executable Python code correctness on a held-out MBPP test set?

## Historical training protocol

The notebook uses Qwen2.5-Coder-14B-Instruct with 4-bit quantization and LoRA.

The executed notebook reports 284 training samples and 257 evaluation samples, with 5 epochs, effective batch size 8, and approximately 180 total update steps.

The 284 training examples are composed of 164 HumanEval examples plus the MBPP train split. Therefore HumanEval is **training data**, not a valid held-out evaluation set.

## Main methodological gap

The notebook's evaluation consists of five hand-written prompts and prints generated answers. It does not run the model against the full MBPP test set with executable tests or report pass@1.

HumanEval should not be presented as a held-out benchmark for this trained model because all 164 HumanEval examples were added to the training data.

## Controlled redesign

This branch evaluates:
- frozen base Qwen2.5-Coder-14B-Instruct
- trained LoRA adapter

on the official MBPP sanitized test split.

Primary metric:
- task pass rate under executable tests

Secondary metrics:
- per-test pass rate
- challenge-test pass rate when available
- generation latency
- parse/compile-style execution failures

Use deterministic decoding for the comparison.

## Hypothesis

> LoRA fine-tuning on coding examples will improve executable MBPP test accuracy relative to the frozen base model.

This is falsifiable.

## Leakage note

HumanEval can still be evaluated as a leakage diagnostic, but any score should be explicitly labeled as non-held-out for this training protocol.