# Reproducibility

Install the evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

Run a smoke test:

```bash
python scripts/evaluate_mbpp.py --mode base --limit 20 --output results/base_20.json
python scripts/evaluate_mbpp.py --mode adapter --adapter ./qwen-coder-final --limit 20 --output results/adapter_20.json
python scripts/analyze_results.py results/base_20.json results/adapter_20.json
```

For the primary evaluation, remove `--limit`.

Keep the same prompt, max tokens, decoding settings, dataset revision and test executor for both conditions.