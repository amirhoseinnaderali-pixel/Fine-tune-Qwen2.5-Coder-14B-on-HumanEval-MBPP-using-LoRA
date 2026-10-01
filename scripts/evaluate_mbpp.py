import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def extract_code(text):
    text = (text or "").strip()
    fence = chr(96) * 3
    pf = fence + "python"
    if pf in text:
        return text.split(pf, 1)[1].split(fence, 1)[0].strip()
    if fence in text:
        parts = text.split(fence)
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def execute_candidate(code, setup_code, tests, timeout=8):
    import subprocess

    program = (setup_code or "") + "\n" + code + "\n" + "\n".join(tests)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python3", "-c", program],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "passed": proc.returncode == 0,
            "stderr": proc.stderr,
            "timeout": False,
            "seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "stderr": (exc.stderr or "") + "\nExecution timeout",
            "timeout": True,
            "seconds": time.perf_counter() - started,
        }


def load_model(base_model, mode, adapter):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if mode == "base":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
            is_trainable=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    model.eval()
    return model, tokenizer


def make_prompt(tokenizer, description):
    system = (
        "You are an expert Python programmer. "
        "Return only a complete solution. "
        "Do not include explanations outside the code."
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": description},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    prompt = (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + description + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def evaluate(args):
    dataset = load_dataset(args.dataset, "sanitized", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model, tokenizer = load_model(args.base_model, args.mode, args.adapter)
    device = next(model.parameters()).device

    task_passed = 0
    test_passed = 0
    test_total = 0
    challenge_passed = 0
    challenge_total = 0
    rows = []

    for example in tqdm(dataset, desc=args.mode):
        inputs = make_prompt(tokenizer, example["text"]).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_seconds = time.perf_counter() - started

        generated = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        )
        code = extract_code(generated)

        tests = list(example.get("test_list") or [])
        challenge = list(example.get("challenge_test_list") or [])
        setup = example.get("test_setup_code", "") or ""

        test_result = execute_candidate(code, setup, tests, timeout=args.timeout)
        challenge_result = execute_candidate(
            code, setup, challenge, timeout=args.timeout
        ) if challenge else {"passed": None, "timeout": False, "seconds": 0.0}

        task_ok = test_result["passed"]
        task_passed += int(task_ok)
        test_passed += int(task_ok)
        test_total += 1
        if challenge:
            challenge_total += 1
            challenge_passed += int(challenge_result["passed"])

        rows.append(
            {
                "task_id": example.get("task_id"),
                "code": code,
                "task_passed": task_ok,
                "test_list_present": bool(tests),
                "challenge_present": bool(challenge),
                "challenge_passed": challenge_result["passed"],
                "generation_seconds": generation_seconds,
                "execution_stderr": test_result["stderr"],
                "timeout": test_result["timeout"],
            }
        )

    n = len(rows)
    result = {
        "mode": args.mode,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": "test",
        "examples": n,
        "task_pass_rate": task_passed / n if n else 0.0,
        "test_pass_rate": test_passed / test_total if test_total else 0.0,
        "challenge_task_rate": challenge_passed / challenge_total if challenge_total else None,
        "rows": rows,
    }

    out = Path(args.output or f"results/{args.mode}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "adapter"], required=True)
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.mode == "adapter" and not args.adapter:
        parser.error("--adapter is required for adapter mode")
    evaluate(args)
