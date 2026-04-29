#!/usr/bin/env python3
"""
Sampling Ratchet Effect Experiment
===================================
Test whether the Ratchet Effect holds under temperature > 0 sampling.

Two tests:
  Test A (Same-seed): T=0.7, same seed at B1=256 and B2=512
    → Natural-stop non-recovery should hold (deterministic generation)
    → Proves Ratchet extends beyond greedy to any deterministic decoding

  Test B (Cross-seed): T=0.7, different seeds at B1=256 and B2=512
    → Natural-stop may "recover" due to fresh attempt
    → Tests whether BUDGET ASYMMETRY persists under stochastic generation

Usage: CUDA_VISIBLE_DEVICES=1 python3 -u run_sampling_ratchet.py
"""
import sys, os, json, time, re, gc, random, argparse
from pathlib import Path
from datetime import datetime
import torch, numpy as np

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR = BASE / "results_v2" / "sampling_ratchet"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
N_SAMPLES = 200
TEMPERATURE = 0.7
TOP_P = 0.9
B1 = 256
B2 = 512
MODEL_PATH = "/mnt/data/pre_model/Qwen2.5-7B-Instruct"
DATA_FILE_MATH = BASE / "math_real_200.json"

def extract_ans(text):
    """Extract answer from model output."""
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        if nums: return nums[-1]
    for pat in [
        r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+|so the answer is|final answer[:\s]+)([^\n.]+)',
        r'answer[:\s]+([^\n.]+)',
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
            if nums: return nums[-1]
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line: continue
        m = re.search(r'(?:is|=|equals?)\s*\$?(-?\d+\.?\d*)', line, re.IGNORECASE)
        if m: return m.group(1)
        nums = re.findall(r'-?\d+\.?\d*', line)
        if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]

def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print(f"Sampling Ratchet Effect | {datetime.now()}", flush=True)
    print(f"Model: Qwen2.5-7B-Instruct | T={TEMPERATURE} top_p={TOP_P}", flush=True)
    print(f"Budgets: {B1}→{B2} | Seeds: {SEEDS}", flush=True)

    device = "cuda:0"
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = "cuda:0"

    # Load data
    if DATA_FILE_MATH.exists():
        with open(DATA_FILE_MATH) as f:
            data = json.load(f)
    else:
        # Try alternate path
        alt = BASE.parent / "tts_bench_optimized" / "math_real_200.json"
        if alt.exists():
            with open(alt) as f:
                data = json.load(f)
        else:
            print(f"ERROR: Cannot find math_real_200.json", flush=True)
            return

    data = data[:N_SAMPLES]
    print(f"Loaded {len(data)} MATH samples", flush=True)

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

    all_results = {}

    # ================================================================
    # Generate at all seed × budget combinations
    # ================================================================
    combos = [(seed, budget) for seed in SEEDS for budget in [B1, B2]]

    for seed, budget in combos:
        key = f"seed{seed}_b{budget}"
        ckpt_file = CKPT_DIR / f"{key}.json"

        if ckpt_file.exists():
            print(f"  Loading cached: {key}", flush=True)
            with open(ckpt_file) as f:
                all_results[key] = json.load(f)
            continue

        print(f"\n=== Generating: seed={seed}, budget={budget} ===", flush=True)
        set_seed(seed)
        samples = []

        for i in range(N_SAMPLES):
            q = data[i]["query"]
            gt = str(data[i].get("ground_truth", data[i].get("answer", "")))

            try:
                content = q + "\nLet's think step by step."
                messages = [{"role": "user", "content": content}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = tok(text, return_tensors="pt").to(device)

                # Reset seed PER SAMPLE for reproducibility
                set_seed(seed + i * 10000)

                with torch.no_grad():
                    out = model.generate(
                        **inp, max_new_tokens=budget,
                        do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                        pad_token_id=tok.eos_token_id,
                    )

                gen_tok = out.shape[1] - inp["input_ids"].shape[1]
                gen_ids = out[0][inp["input_ids"].shape[1]:].tolist()
                full_text = tok.decode(gen_ids, skip_special_tokens=True)

                truncated = gen_tok >= budget - 5
                ans = extract_ans(full_text)
                ok = check(ans, gt)

                samples.append({
                    "q": i, "gt": gt, "ans": ans, "ok": ok,
                    "gen_tokens": gen_tok, "truncated": truncated,
                })
            except Exception as e:
                samples.append({"q": i, "gt": gt, "err": str(e), "ok": False,
                               "gen_tokens": 0, "truncated": False})

            if (i + 1) % 50 == 0:
                acc = sum(s["ok"] for s in samples) / len(samples) * 100
                trunc = sum(s["truncated"] for s in samples) / len(samples) * 100
                print(f"  [{i+1}/{N_SAMPLES}] acc={acc:.1f}% trunc={trunc:.0f}%", flush=True)

        # Save checkpoint
        acc = sum(s["ok"] for s in samples) / len(samples) * 100
        trunc = sum(s["truncated"] for s in samples) / len(samples) * 100
        result = {
            "metadata": {
                "model": "Qwen2.5-7B-Instruct", "dataset": "MATH",
                "temperature": TEMPERATURE, "top_p": TOP_P,
                "seed": seed, "budget": budget, "method": "sampling",
                "n": N_SAMPLES, "accuracy": acc, "truncation_rate": trunc,
            },
            "results": samples,
        }
        all_results[key] = result
        with open(ckpt_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {key} | acc={acc:.1f}% trunc={trunc:.0f}%", flush=True)

    # ================================================================
    # Analysis
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"SAMPLING RATCHET EFFECT ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    # Test A: Same-seed analysis
    print(f"\n--- Test A: Same-seed recovery (T={TEMPERATURE}) ---", flush=True)
    print(f"{'Seed':>6} {'NatW':>6} {'NatR':>6} {'CeilW':>6} {'CeilR':>6} {'Ceil%':>7}", flush=True)

    for seed in SEEDS:
        r_b1 = all_results[f"seed{seed}_b{B1}"]["results"]
        r_b2 = all_results[f"seed{seed}_b{B2}"]["results"]

        nat_w, nat_r = 0, 0
        ceil_w, ceil_r = 0, 0

        for s1, s2 in zip(r_b1, r_b2):
            if s1.get("err") or s2.get("err"): continue
            if s1["ok"]: continue  # Only look at wrong at B1

            if s1["truncated"]:
                ceil_w += 1
                if s2["ok"]: ceil_r += 1
            else:
                nat_w += 1
                if s2["ok"]: nat_r += 1

        ceil_pct = (ceil_r / ceil_w * 100) if ceil_w > 0 else 0
        print(f"{seed:>6} {nat_w:>6} {nat_r:>6} {ceil_w:>6} {ceil_r:>6} {ceil_pct:>6.1f}%", flush=True)

    # Test B: Cross-seed analysis (B1 with seed S, B2 with different seed)
    print(f"\n--- Test B: Cross-seed recovery (T={TEMPERATURE}) ---", flush=True)
    print(f"{'B1_seed':>8} {'B2_seed':>8} {'NatW':>6} {'NatR':>6} {'NatR%':>7} {'CeilW':>6} {'CeilR':>6} {'CeilR%':>7}", flush=True)

    cross_results = []
    for i, s1 in enumerate(SEEDS):
        for j, s2 in enumerate(SEEDS):
            if s1 == s2: continue  # Skip same-seed (already in Test A)

            r_b1 = all_results[f"seed{s1}_b{B1}"]["results"]
            r_b2 = all_results[f"seed{s2}_b{B2}"]["results"]

            nat_w, nat_r = 0, 0
            ceil_w, ceil_r = 0, 0

            for sam1, sam2 in zip(r_b1, r_b2):
                if sam1.get("err") or sam2.get("err"): continue
                if sam1["ok"]: continue

                if sam1["truncated"]:
                    ceil_w += 1
                    if sam2["ok"]: ceil_r += 1
                else:
                    nat_w += 1
                    if sam2["ok"]: nat_r += 1

            nat_pct = (nat_r / nat_w * 100) if nat_w > 0 else 0
            ceil_pct = (ceil_r / ceil_w * 100) if ceil_w > 0 else 0
            print(f"{s1:>8} {s2:>8} {nat_w:>6} {nat_r:>6} {nat_pct:>6.1f}% {ceil_w:>6} {ceil_r:>6} {ceil_pct:>6.1f}%", flush=True)
            cross_results.append((nat_w, nat_r, ceil_w, ceil_r))

    # Aggregate cross-seed
    tot_nat_w = sum(r[0] for r in cross_results)
    tot_nat_r = sum(r[1] for r in cross_results)
    tot_ceil_w = sum(r[2] for r in cross_results)
    tot_ceil_r = sum(r[3] for r in cross_results)
    print(f"\n{'TOTAL':>8} {'':>8} {tot_nat_w:>6} {tot_nat_r:>6} {tot_nat_r/tot_nat_w*100 if tot_nat_w else 0:>6.1f}% {tot_ceil_w:>6} {tot_ceil_r:>6} {tot_ceil_r/tot_ceil_w*100 if tot_ceil_w else 0:>6.1f}%", flush=True)

    # Also: baseline at @256 with same seed but @256 with diff seed
    # → This shows how much of "recovery" is just sampling variation vs budget effect
    print(f"\n--- Test C: Budget-independent sampling variation ---", flush=True)
    print(f"{'B1_seed':>8} {'B2_seed':>8} {'NatW':>6} {'NatR@256':>9} {'CeilW':>6} {'CeilR@256':>10}", flush=True)
    print(f"{'':>8} {'':>8} {'':>6} {'(fresh)':>9} {'':>6} {'(fresh)':>10}", flush=True)

    for s1 in SEEDS:
        for s2 in SEEDS:
            if s1 == s2: continue
            r_b1 = all_results[f"seed{s1}_b{B1}"]["results"]
            r_b2 = all_results[f"seed{s2}_b{B1}"]["results"]  # SAME budget, diff seed

            nat_w, nat_r = 0, 0
            ceil_w, ceil_r = 0, 0
            for sam1, sam2 in zip(r_b1, r_b2):
                if sam1.get("err") or sam2.get("err"): continue
                if sam1["ok"]: continue
                if sam1["truncated"]:
                    ceil_w += 1
                    if sam2["ok"]: ceil_r += 1
                else:
                    nat_w += 1
                    if sam2["ok"]: nat_r += 1
            nat_pct = (nat_r / nat_w * 100) if nat_w > 0 else 0
            ceil_pct = (ceil_r / ceil_w * 100) if ceil_w > 0 else 0
            print(f"{s1:>8} {s2:>8} {nat_w:>6} {nat_r:>6}({nat_pct:.0f}%) {ceil_w:>6} {ceil_r:>6}({ceil_pct:.0f}%)", flush=True)

    # ================================================================
    # Summary for paper
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY FOR REBUTTAL", flush=True)
    print(f"{'='*70}", flush=True)

    # Test A summary
    for seed in SEEDS:
        r_b1 = all_results[f"seed{seed}_b{B1}"]["results"]
        r_b2 = all_results[f"seed{seed}_b{B2}"]["results"]
        nat_w, nat_r = 0, 0
        for s1, s2 in zip(r_b1, r_b2):
            if s1.get("err") or s2.get("err") or s1["ok"]: continue
            if not s1["truncated"]:
                nat_w += 1
                if s2["ok"]: nat_r += 1
        print(f"Test A (same-seed T={TEMPERATURE}): seed={seed} → nat_stop recovery: {nat_r}/{nat_w}", flush=True)

    print(flush=True)

    # Test B summary
    print(f"Test B (cross-seed T={TEMPERATURE}): pooled → nat_stop recovery: {tot_nat_r}/{tot_nat_w} ({tot_nat_r/tot_nat_w*100:.1f}%), ceil recovery: {tot_ceil_r}/{tot_ceil_w} ({tot_ceil_r/tot_ceil_w*100:.1f}%)", flush=True)

    # Compare with greedy baseline
    greedy_file = CKPT_DIR.parent / "Qwen2.5-7B_MATH_baseline_256_s123_verify.json"
    if not greedy_file.exists():
        greedy_file = Path('/mnt/data2/zcz/tts_bench_optimized/results_v2/Qwen2.5-7B_MATH_baseline_256_s123_verify.json')
    if greedy_file.exists():
        d = json.load(open(greedy_file))
        acc = sum(r.get('ok',False) for r in d['results'])/len(d['results'])*100
        trunc = sum(r.get('truncated',False) for r in d['results'])/len(d['results'])*100
        print(f"\nGreedy baseline @256: acc={acc:.1f}% trunc={trunc:.0f}%", flush=True)

    # Print per-condition accuracy for sanity check
    print(f"\nPer-condition summary:", flush=True)
    for seed in SEEDS:
        for budget in [B1, B2]:
            key = f"seed{seed}_b{budget}"
            r = all_results[key]
            m = r["metadata"]
            print(f"  seed={seed} @{budget}: acc={m['accuracy']:.1f}% trunc={m['truncation_rate']:.0f}%", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
