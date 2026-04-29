#!/usr/bin/env python3
"""
HumanEval Code Domain Experiment
==================================
Adds code generation domain to the Truncation Signature paper.
Tests whether Ratchet Effect + Token Signal extend to code tasks.

Model: Qwen2.5-7B-Instruct
Dataset: HumanEval (164 Python problems)
Method: Generate @1024, extract virtual budgets (greedy decoding)
Evaluation: Execute generated code against test cases

Usage: CUDA_VISIBLE_DEVICES=1 python3 -u run_humaneval.py
"""
import sys, os, json, time, re, gc, traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch, numpy as np

BASE = Path(__file__).parent
CKPT_DIR = BASE / "results_v2" / "humaneval"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/mnt/data/pre_model/Qwen2.5-7B-Instruct"
MAX_BUDGET = 1024  # Generate once at @1024, truncate for lower budgets
BUDGETS = [128, 256, 384, 512, 768, 1024]
HUMANEVAL_FILE = BASE / "humaneval_164.json"

def extract_code_completion(output: str, prompt: str) -> str:
    """Extract the code completion from model output."""
    # The model continues from the prompt; extract the completion
    text = output.strip()
    # Remove any trailing chat tokens
    text = re.sub(r'<\|im_end\|>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|endoftext\|>.*', '', text, flags=re.DOTALL)
    return text

def evaluate_code(problem: dict, completion: str) -> bool:
    """Execute generated code against test cases. Returns True if passes."""
    task_id = problem['task_id']
    prompt = problem['prompt']
    test = problem['test']
    entry_point = problem['entry_point']

    # Construct the full code: prompt + completion + test + check call
    full_code = prompt + completion + '\n' + test + '\n'
    full_code += f'check({entry_point})\n'

    try:
        # Execute in a restricted namespace
        exec_globals = {}
        exec(full_code, exec_globals)
        return True
    except Exception as e:
        return False

def extract_code_from_chat(output: str) -> str:
    """Extract code block from chat-style output if model wraps in ```."""
    # Try to extract code from ```python ... ``` blocks
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    return output

def main():
    print(f"HumanEval Experiment | {datetime.now()}", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"Budgets: {BUDGETS} (virtual extraction from @{MAX_BUDGET})", flush=True)

    device = "cuda:0"

    # Load HumanEval
    with open(HUMANEVAL_FILE) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} HumanEval problems", flush=True)

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    # ================================================================
    # Step 1: Generate at @MAX_BUDGET
    # ================================================================
    ckpt_file = CKPT_DIR / "humaneval_1024_raw.json"

    if ckpt_file.exists():
        print(f"Loading cached generations: {ckpt_file}", flush=True)
        with open(ckpt_file) as f:
            raw_results = json.load(f)
    else:
        print(f"\n=== Generating at @{MAX_BUDGET} ===", flush=True)
        raw_results = []

        for i, prob in enumerate(problems):
            task_id = prob['task_id']
            prompt = prob['prompt']

            # Format as completion task (not chat) for HumanEval
            # Use the prompt directly — the model should complete the function
            inp = tok(prompt, return_tensors="pt").to(device)

            try:
                with torch.no_grad():
                    out = model.generate(
                        **inp, max_new_tokens=MAX_BUDGET,
                        do_sample=False,  # Greedy
                        pad_token_id=tok.eos_token_id,
                        temperature=1.0,
                    )

                gen_ids = out[0][inp["input_ids"].shape[1]:].tolist()
                gen_text = tok.decode(gen_ids, skip_special_tokens=True)
                gen_tokens = len(gen_ids)

            except Exception as e:
                gen_text = ""
                gen_tokens = 0
                print(f"  ERROR {task_id}: {e}", flush=True)

            raw_results.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": gen_text,
                "gen_tokens": gen_tokens,
                "test": prob['test'],
                "entry_point": prob['entry_point'],
                "canonical_solution": prob.get('canonical_solution', ''),
            })

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(problems)}] avg_tokens={np.mean([r['gen_tokens'] for r in raw_results]):.0f}", flush=True)

        # Save raw results
        with open(ckpt_file, 'w') as f:
            json.dump(raw_results, f, indent=2)
        print(f"Saved {len(raw_results)} raw results", flush=True)

    # ================================================================
    # Step 2: Virtual budget extraction + evaluation
    # ================================================================
    print(f"\n=== Virtual Budget Extraction + Evaluation ===", flush=True)

    budget_results = {}  # budget -> list of {task_id, ok, gen_tokens, truncated}

    for budget in BUDGETS:
        budget_key = f"b{budget}"
        bfile = CKPT_DIR / f"humaneval_{budget_key}.json"

        if bfile.exists():
            print(f"  Loading cached: {budget_key}", flush=True)
            with open(bfile) as f:
                budget_results[budget] = json.load(f)
            continue

        samples = []
        for raw in raw_results:
            task_id = raw['task_id']
            full_completion = raw['completion']
            total_tokens = raw['gen_tokens']

            # Truncate at budget boundary
            if total_tokens <= budget:
                # Output fits within budget — natural stop
                completion = full_completion
                gen_tokens = total_tokens
                truncated = False
            else:
                # Truncate: take only first 'budget' tokens
                # Re-tokenize the truncated text
                tokens_so_far = tok.encode(full_completion)
                if len(tokens_so_far) > budget:
                    truncated_text = tok.decode(tokens_so_far[:budget])
                else:
                    truncated_text = full_completion
                completion = truncated_text
                gen_tokens = min(total_tokens, budget)
                truncated = gen_tokens >= budget - 5

            # Evaluate
            # Build the full problem dict for evaluation
            prob = {
                'task_id': task_id,
                'prompt': raw['prompt'],
                'test': raw['test'],
                'entry_point': raw['entry_point'],
            }
            ok = evaluate_code(prob, completion)

            samples.append({
                "task_id": task_id,
                "ok": ok,
                "gen_tokens": gen_tokens,
                "truncated": truncated,
                "completion_preview": completion[:100],
            })

        # Save
        acc = sum(s['ok'] for s in samples) / len(samples) * 100
        trunc_rate = sum(s['truncated'] for s in samples) / len(samples) * 100
        budget_results[budget] = {
            "metadata": {
                "model": "Qwen2.5-7B-Instruct",
                "dataset": "HumanEval",
                "budget": budget,
                "method": "virtual_budget",
                "n": len(samples),
                "accuracy": acc,
                "truncation_rate": trunc_rate,
            },
            "results": samples,
        }
        with open(bfile, 'w') as f:
            json.dump(budget_results[budget], f, indent=2)
        print(f"  {budget_key}: acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)

    # ================================================================
    # Step 3: Ratchet Effect Analysis
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"RATCHET EFFECT ANALYSIS (HumanEval)", flush=True)
    print(f"{'='*70}", flush=True)

    # Find the most informative transition (most natural-stop errors)
    best_transition = None
    best_nat_count = 0

    for i in range(len(BUDGETS) - 1):
        b1, b2 = BUDGETS[i], BUDGETS[i+1]
        r1 = budget_results[b1]['results']
        r2 = budget_results[b2]['results']

        nat_w = sum(1 for s1 in r1 if not s1['ok'] and not s1['truncated'])
        if nat_w > best_nat_count:
            best_nat_count = nat_w
            best_transition = (b1, b2)

    print(f"Best transition: {best_transition} (nat_stop={best_nat_count})", flush=True)

    # Analyze all transitions
    print(f"\n{'B1':>5} {'B2':>5} {'NatW':>5} {'NatR':>5} {'CeilW':>5} {'CeilR':>5} {'CeilR%':>7}", flush=True)

    total_nat_w, total_nat_r = 0, 0
    total_ceil_w, total_ceil_r = 0, 0

    for i in range(len(BUDGETS) - 1):
        b1, b2 = BUDGETS[i], BUDGETS[i+1]
        r1 = budget_results[b1]['results']
        r2 = budget_results[b2]['results']

        nat_w, nat_r = 0, 0
        ceil_w, ceil_r = 0, 0

        for s1, s2 in zip(r1, r2):
            if s1['ok']: continue  # Only look at wrong at B1
            if s1['truncated']:
                ceil_w += 1
                if s2['ok']: ceil_r += 1
            else:
                nat_w += 1
                if s2['ok']: nat_r += 1

        total_nat_w += nat_w
        total_nat_r += nat_r
        total_ceil_w += ceil_w
        total_ceil_r += ceil_r

        ceil_pct = (ceil_r / ceil_w * 100) if ceil_w > 0 else 0
        print(f"{b1:>5} {b2:>5} {nat_w:>5} {nat_r:>5} {ceil_w:>5} {ceil_r:>5} {ceil_pct:>6.1f}%", flush=True)

    print(f"\nTOTAL: nat={total_nat_r}/{total_nat_w} ({total_nat_r/total_nat_w*100 if total_nat_w else 0:.1f}%), "
          f"ceil={total_ceil_r}/{total_ceil_w} ({total_ceil_r/total_ceil_w*100 if total_ceil_w else 0:.1f}%)", flush=True)

    # ================================================================
    # Step 4: Token Signal AUC
    # ================================================================
    print(f"\n=== Token Signal AUC ===", flush=True)

    from sklearn.metrics import roc_auc_score

    for budget in [256, 512]:
        if budget not in budget_results:
            continue
        samples = budget_results[budget]['results']
        labels = np.array([1 if s['ok'] else 0 for s in samples])
        token_counts = np.array([s['gen_tokens'] for s in samples])

        if len(np.unique(labels)) < 2:
            print(f"  @{budget}: All same label, skip AUC", flush=True)
            continue

        # Token count AUC (higher count → more likely correct)
        # Actually, for token signal, we want: is token count ≈ budget?
        # Use 1 - normalized token count (closer to budget = more likely wrong due to truncation)
        token_signal = token_counts  # Higher = more tokens = more likely correct
        token_auc = roc_auc_score(labels, token_signal)

        # Inverse: truncation signal (token count ≈ budget → wrong)
        trunc_signal = np.abs(token_counts - budget)  # Lower = closer to budget = more likely wrong
        trunc_auc = roc_auc_score(labels, -token_signal)  # Negative: fewer tokens → wrong

        # Actually, the right way: token count as predictor of CORRECTNESS
        # High token count = model wrote more = could be correct OR truncated
        # Let's compute it both ways

        # Standard: token count predicts wrong (high count at budget = truncated = likely wrong)
        wrong_labels = 1 - labels  # 1 = wrong
        token_auc_wrong = roc_auc_score(wrong_labels, token_counts)

        print(f"  @{budget}: Token AUC (predict wrong) = {token_auc_wrong:.3f}, "
              f"Token AUC (predict correct) = {1-token_auc_wrong:.3f}", flush=True)

    # ================================================================
    # Step 5: Budget Spectrum
    # ================================================================
    print(f"\n=== Budget Spectrum ===", flush=True)
    print(f"{'Budget':>7} {'Acc':>7} {'Ceil%':>7} {'Zone':>8}", flush=True)

    for budget in BUDGETS:
        meta = budget_results[budget]['metadata']
        acc = meta['accuracy']
        ceil = meta['truncation_rate']

        if ceil > 50: zone = "TRUNC"
        elif ceil > 10: zone = "INFLECT"
        else: zone = "EFF"

        print(f"{budget:>7} {acc:>6.1f}% {ceil:>6.0f}% {zone:>8}", flush=True)

    # ================================================================
    # Summary for paper
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY FOR PAPER", flush=True)
    print(f"{'='*70}", flush=True)

    # Best transition Ratchet data
    if best_transition:
        b1, b2 = best_transition
        r1 = budget_results[b1]['results']
        r2 = budget_results[b2]['results']
        nat_w = sum(1 for s1, s2 in zip(r1, r2) if not s1['ok'] and not s1['truncated'] and s2['ok'])
        nat_total = sum(1 for s1 in r1 if not s1['ok'] and not s1['truncated'])
        ceil_w = sum(1 for s1 in r1 if not s1['ok'] and s1['truncated'])
        ceil_r = sum(1 for s1, s2 in zip(r1, r2) if not s1['ok'] and s1['truncated'] and s2['ok'])

        print(f"Ratchet @{b1}→@{b2}: nat={nat_w}/{nat_total} ({nat_w/nat_total*100 if nat_total else 0:.1f}%), "
              f"ceil={ceil_r}/{ceil_w} ({ceil_r/ceil_w*100 if ceil_w else 0:.1f}%)", flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
