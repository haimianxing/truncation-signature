#!/usr/bin/env python3
"""
Prospective Validation Experiment
==================================
50 held-out MATH samples (not in the 200 used for paper analysis).
Tests the Three-Zone heuristic prospectively:
  1. Run all 50 at @256 (Phase 1)
  2. Extend only ceiling-hit outputs to @512 (Phase 2)
  3. Compare against baseline: all 50 at @512

Measures: compute savings (total tokens) and accuracy.

Usage: CUDA_VISIBLE_DEVICES=1 python3 -u run_prospective.py
"""
import sys, os, json, time, re, gc, traceback
from pathlib import Path
from datetime import datetime
import torch, numpy as np

BASE = Path(__file__).parent
CKPT_DIR = BASE / "results_v2" / "prospective"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/mnt/data/pre_model/Qwen2.5-7B-Instruct"
B1, B2 = 256, 512
N_HELDOUT = 50

def extract_answer(text):
    """Extract final numerical answer."""
    # Try \boxed{} first
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    # Try "the answer is" pattern
    m = re.search(r'(?:the answer is|therefore[,\s]*(?:the answer is)?)[\s]*([-$\d\.,/]+)', text, re.I)
    if m:
        return m.group(1).strip()
    # Last number
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return text.strip()

def check_answer(pred, gold):
    """Check if predicted answer matches gold."""
    pred = pred.strip().replace(',', '').replace('$', '')
    gold = gold.strip().replace(',', '').replace('$', '')
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except:
        return pred == gold

def main():
    print(f"Prospective Validation | {datetime.now()}", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"B1={B1}, B2={B2}, N={N_HELDOUT}", flush=True)

    device = "cuda:0"

    # Load held-out MATH problems
    with open('/mnt/data2/zcz/tts_bench_optimized/math_test.json') as f:
        all_heldout = json.load(f)
    problems = all_heldout[:N_HELDOUT]
    print(f"Loaded {len(problems)} held-out MATH problems", flush=True)

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
    print(f"Model loaded", flush=True)

    def generate(prompt, max_tokens):
        """Generate with greedy decoding."""
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][inp["input_ids"].shape[1]:].tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        return gen_text, len(gen_ids)

    # ================================================================
    # Phase 1: Run all 50 at @B1=256
    # ================================================================
    phase1_file = CKPT_DIR / "phase1_b256.json"
    if phase1_file.exists():
        print(f"Loading cached Phase 1: {phase1_file}", flush=True)
        with open(phase1_file) as f:
            phase1 = json.load(f)
    else:
        print(f"\n=== Phase 1: @256 (all {N_HELDOUT} samples) ===", flush=True)
        phase1 = []
        for i, prob in enumerate(problems):
            try:
                text, ntok = generate(prob['query'], B1)
                pred = extract_answer(text)
                ok = check_answer(pred, prob['ground_truth'])
                ceiling = ntok >= B1 - 5
                phase1.append({
                    "idx": i, "query": prob['query'][:80],
                    "ground_truth": prob['ground_truth'],
                    "pred": pred, "ok": ok,
                    "gen_tokens": ntok, "ceiling_hit": ceiling,
                    "full_output": text,
                })
            except Exception as e:
                phase1.append({
                    "idx": i, "query": prob['query'][:80],
                    "ground_truth": prob['ground_truth'],
                    "pred": "ERROR", "ok": False,
                    "gen_tokens": 0, "ceiling_hit": False,
                    "full_output": str(e),
                })
                print(f"  ERROR {i}: {e}", flush=True)

            if (i+1) % 10 == 0:
                acc = sum(r['ok'] for r in phase1) / len(phase1) * 100
                ceil = sum(r['ceiling_hit'] for r in phase1) / len(phase1) * 100
                print(f"  [{i+1}/{N_HELDOUT}] acc={acc:.1f}% ceil={ceil:.1f}%", flush=True)

        with open(phase1_file, 'w') as f:
            json.dump(phase1, f, indent=2)

    p1_acc = sum(r['ok'] for r in phase1) / len(phase1) * 100
    p1_ceil = sum(r['ceiling_hit'] for r in phase1) / len(phase1) * 100
    p1_tokens = sum(r['gen_tokens'] for r in phase1)
    n_ceiling = sum(r['ceiling_hit'] for r in phase1)
    n_natstop = sum(not r['ceiling_hit'] for r in phase1)
    n_wrong_natstop = sum(1 for r in phase1 if not r['ok'] and not r['ceiling_hit'])
    n_wrong_ceiling = sum(1 for r in phase1 if not r['ok'] and r['ceiling_hit'])

    print(f"\nPhase 1 Results: acc={p1_acc:.1f}%, ceiling={p1_ceil:.1f}%", flush=True)
    print(f"  NatStop: {n_natstop} (wrong: {n_wrong_natstop})", flush=True)
    print(f"  Ceiling: {n_ceiling} (wrong: {n_wrong_ceiling})", flush=True)

    # ================================================================
    # Phase 2: Extend only ceiling-hit to @512
    # ================================================================
    phase2_file = CKPT_DIR / "phase2_b512_extended.json"
    if phase2_file.exists():
        print(f"Loading cached Phase 2: {phase2_file}", flush=True)
        with open(phase2_file) as f:
            phase2 = json.load(f)
    else:
        print(f"\n=== Phase 2: Extend ceiling-hit to @512 ({n_ceiling} samples) ===", flush=True)
        phase2 = []
        for r1 in phase1:
            if r1['ceiling_hit']:
                try:
                    # Find the original problem
                    idx = r1['idx']
                    text, ntok = generate(problems[idx]['query'], B2)
                    pred = extract_answer(text)
                    ok = check_answer(pred, problems[idx]['ground_truth'])
                    phase2.append({
                        "idx": idx, "pred": pred, "ok": ok,
                        "gen_tokens": ntok,
                    })
                except Exception as e:
                    phase2.append({
                        "idx": r1['idx'], "pred": "ERROR", "ok": False,
                        "gen_tokens": 0,
                    })
                    print(f"  ERROR {idx}: {e}", flush=True)
            else:
                # Natural stop — keep Phase 1 result
                phase2.append({
                    "idx": r1['idx'], "pred": r1['pred'], "ok": r1['ok'],
                    "gen_tokens": r1['gen_tokens'],
                })

        with open(phase2_file, 'w') as f:
            json.dump(phase2, f, indent=2)

    # ================================================================
    # Phase 3: Baseline — all 50 at @512
    # ================================================================
    phase3_file = CKPT_DIR / "phase3_b512_all.json"
    if phase3_file.exists():
        print(f"Loading cached Phase 3: {phase3_file}", flush=True)
        with open(phase3_file) as f:
            phase3 = json.load(f)
    else:
        print(f"\n=== Phase 3: Baseline @512 (all {N_HELDOUT} samples) ===", flush=True)
        phase3 = []
        for i, prob in enumerate(problems):
            try:
                text, ntok = generate(prob['query'], B2)
                pred = extract_answer(text)
                ok = check_answer(pred, prob['ground_truth'])
                phase3.append({
                    "idx": i, "pred": pred, "ok": ok,
                    "gen_tokens": ntok,
                })
            except Exception as e:
                phase3.append({
                    "idx": i, "pred": "ERROR", "ok": False,
                    "gen_tokens": 0,
                })
            if (i+1) % 10 == 0:
                acc = sum(r['ok'] for r in phase3) / len(phase3) * 100
                print(f"  [{i+1}/{N_HELDOUT}] acc={acc:.1f}%", flush=True)

        with open(phase3_file, 'w') as f:
            json.dump(phase3, f, indent=2)

    # ================================================================
    # Analysis
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"PROSPECTIVE VALIDATION RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    p2_acc = sum(r['ok'] for r in phase2) / len(phase2) * 100
    p3_acc = sum(r['ok'] for r in phase3) / len(phase3) * 100

    # Compute tokens for adaptive strategy
    adaptive_tokens = 0
    for r1 in phase1:
        if r1['ceiling_hit']:
            adaptive_tokens += B1  # Phase 1 attempt
            # Find Phase 2 result for this sample
            p2_r = next((r for r in phase2 if r['idx'] == r1['idx']), None)
            if p2_r:
                adaptive_tokens += B2  # Phase 2 full attempt
        else:
            adaptive_tokens += r1['gen_tokens']  # Only used actual tokens

    baseline_tokens = N_HELDOUT * B2  # Baseline: all at @512

    savings = (1 - adaptive_tokens / baseline_tokens) * 100

    print(f"\n--- Accuracy ---", flush=True)
    print(f"  Phase 1 (@256, all):           {p1_acc:.1f}% ({sum(r['ok'] for r in phase1)}/{len(phase1)})", flush=True)
    print(f"  Adaptive (256→512 conditional): {p2_acc:.1f}% ({sum(r['ok'] for r in phase2)}/{len(phase2)})", flush=True)
    print(f"  Baseline (@512, all):           {p3_acc:.1f}% ({sum(r['ok'] for r in phase3)}/{len(phase3)})", flush=True)

    print(f"\n--- Compute ---", flush=True)
    print(f"  Adaptive total tokens: {adaptive_tokens:,}", flush=True)
    print(f"  Baseline total tokens: {baseline_tokens:,}", flush=True)
    print(f"  Compute savings:       {savings:.1f}%", flush=True)

    print(f"\n--- Ratchet Verification ---", flush=True)
    # Check: did any natural-stop wrong answers become correct?
    for r1 in phase1:
        if not r1['ok'] and not r1['ceiling_hit']:
            p3_r = next((r for r in phase3 if r['idx'] == r1['idx']), None)
            if p3_r and p3_r['ok']:
                print(f"  *** NATURAL-STOP RECOVERY: idx={r1['idx']} (rare!)", flush=True)
    nat_wrong = sum(1 for r in phase1 if not r['ok'] and not r['ceiling_hit'])
    nat_recovered = sum(1 for r1 in phase1
                        if not r1['ok'] and not r1['ceiling_hit']
                        and any(r['idx'] == r1['idx'] and r['ok'] for r in phase3))
    print(f"  Natural-stop wrong→correct: {nat_recovered}/{nat_wrong}", flush=True)

    ceil_wrong = sum(1 for r in phase1 if not r['ok'] and r['ceiling_hit'])
    ceil_recovered = sum(1 for r1 in phase1
                         if not r1['ok'] and r1['ceiling_hit']
                         and any(r['idx'] == r1['idx'] and r['ok'] for r in phase3))
    print(f"  Ceiling-hit wrong→correct:  {ceil_recovered}/{ceil_wrong}", flush=True)

    print(f"\n--- Paper-Ready Summary ---", flush=True)
    print(f"  N={N_HELDOUT} held-out MATH samples (not in training set)", flush=True)
    print(f"  Adaptive accuracy: {p2_acc:.1f}% (vs baseline {p3_acc:.1f}%)", flush=True)
    print(f"  Accuracy delta: {p2_acc - p3_acc:+.1f}%", flush=True)
    print(f"  Compute savings: {savings:.1f}%", flush=True)
    print(f"  Ceiling rate at @256: {p1_ceil:.1f}%", flush=True)

    # Save summary
    summary = {
        "n": N_HELDOUT,
        "phase1_acc": p1_acc,
        "adaptive_acc": p2_acc,
        "baseline_acc": p3_acc,
        "accuracy_delta": p2_acc - p3_acc,
        "adaptive_tokens": adaptive_tokens,
        "baseline_tokens": baseline_tokens,
        "compute_savings_pct": savings,
        "ceiling_rate_256": p1_ceil,
        "nat_recovered": nat_recovered,
        "nat_wrong": nat_wrong,
        "ceil_recovered": ceil_recovered,
        "ceil_wrong": ceil_wrong,
    }
    with open(CKPT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {CKPT_DIR / 'summary.json'}", flush=True)
    print(f"Done: {datetime.now()}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
