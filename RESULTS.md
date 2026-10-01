# Results

No full controlled MBPP execution benchmark has been run in this branch.

## Historical facts

- 164 HumanEval examples were included in training.
- MBPP test contains 257 examples in the checked-in notebook environment.
- The notebook trained for 5 epochs and reported ~180 steps.
- The notebook did not compute executable pass@1 over the 257-example test set.

## Required comparison

| Model | Split | Metric |
|---|---|---|
| Base Qwen2.5-Coder-14B-Instruct | MBPP sanitized test | executable task pass rate |
| LoRA adapter | same | executable task pass rate |

HumanEval should not be used as the primary held-out result for this experiment.