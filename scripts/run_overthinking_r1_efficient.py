#!/usr/bin/env python3
"""
Overthinking Bridge Experiment — OPTIMIZED VERSION
DeepSeek-R1-Distill-Qwen-7B on GPU 6

Key optimization: Generate at max budget (@8192), then extract virtual
budgets by truncation. With greedy decoding, first N tokens of @8192
generation = exact same as @N generation. One run gives ALL budgets.

This is scientifically cleaner: same generation, different truncation points.
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
N_SAMPLES = 200
MAX_BUDGET = 8192
BASE = Path(__file__).parent

# Virtual budgets to extract from each full generation
VIRTUAL_BUDGETS = [256, 512, 1024, 2048, 4096, 8192]

def extract_ans(text, ds="MATH"):
    if "####" in text:
        nums = re.findall(r'-?\d+\.?\d*', text.split("####")[-1])
        return nums[-1] if nums else text.strip()[-50:]
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
        return nums[-1] if nums else boxed[-1]
    for pat in [r'(?:therefore|thus|the answer is)[:\s]+([^\n.]+)',
                r'answer[:\s]+([^\n.]+)', r'=\s*([+-]?\d+\.?\d*)\s*$']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            nums = re.findall(r'-?\d+\.?\d*', m.group(1) if 'answer' in pat.lower() or 'thus' in pat.lower() else m.group(0))
            if nums: return nums[-1]
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()[-50:]

def check(p, g):
    p, g = p.strip().replace(',','').replace(' ',''), str(g).strip().replace(',','').replace(' ','')
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return p.lower() == g.lower()

def main():
    device = "cuda:0"
    print(f"Overthinking Bridge — EFFICIENT VERSION | {datetime.now()}", flush=True)
    print(f"Strategy: Generate @{MAX_BUDGET}, extract virtual budgets via truncation", flush=True)

    model_path = "/mnt/data/pre_model/DeepSeek-R1-Distill-Qwen-7B"
    data_file = BASE / "math_real_200.json"
    ckpt_dir = BASE / "results_overthinking"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Loading {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Model loaded, {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    with open(data_file) as f:
        alldata = json.load(f)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = alldata[:N_SAMPLES]

    # Load checkpoint
    ckpt_file = ckpt_dir / "overthinking_r1_7b_math_efficient.json"
    all_results = {}
    raw_outputs = {}  # Store raw token sequences for virtual budget extraction

    if ckpt_file.exists():
        with open(ckpt_file) as f:
            saved = json.load(f)
        all_results = saved.get("results", {})
        raw_outputs = saved.get("raw_outputs", {})
        # raw_outputs stores token counts and answer positions
        print(f"Loaded checkpoint: {len(raw_outputs)} samples already processed", flush=True)

    # Phase 1: Generate all samples at MAX_BUDGET
    print(f"\n=== Phase 1: Generate {N_SAMPLES} samples @{MAX_BUDGET} ===", flush=True)
    samples_full = []

    start_idx = len(raw_outputs)
    if start_idx > 0:
        print(f"  Resuming from sample {start_idx}", flush=True)

    for i in range(start_idx, N_SAMPLES):
        q = data[i]["query"]
        gt = str(data[i].get("ground_truth", ""))

        try:
            content = f"{q}\n\nPlease think step by step, then give your final numerical answer after ####."
            messages = [{"role": "user", "content": content}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=MAX_BUDGET,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=tok.eos_token_id,
                )

            gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
            full_response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

            # Extract answer from full response
            full_ans = extract_ans(full_response, "MATH")
            full_ok = check(full_ans, gt)

            # Find where #### appears (answer position)
            ans_pos = full_response.find("####")
            has_boxed = "\\boxed{" in full_response

            # Store generated token IDs for precise virtual budget extraction
            gen_ids = out[0][inp["input_ids"].shape[1]:].tolist()

            sample_info = {
                "q": i, "gt": gt,
                "gen_tokens": gen_tokens,
                "hit_ceiling": gen_tokens >= MAX_BUDGET - 5,
                "full_ans": full_ans,
                "full_ok": full_ok,
                "gen_ids": gen_ids,  # Token-level IDs for precise truncation
            }

            raw_outputs[str(i)] = sample_info

        except Exception as e:
            raw_outputs[str(i)] = {
                "q": i, "gt": gt, "err": str(e),
                "gen_tokens": 0, "hit_ceiling": False,
            }

        # Progress and checkpoint every 10 samples (more frequent for long-running)
        done_count = len(raw_outputs)
        if (done_count) % 10 == 0 or done_count == N_SAMPLES:
            ok_count = sum(1 for v in raw_outputs.values() if v.get("full_ok"))
            ceil_count = sum(1 for v in raw_outputs.values() if v.get("hit_ceiling"))
            avg_tok = sum(v.get("gen_tokens", 0) for v in raw_outputs.values()) / len(raw_outputs)
            elapsed = time.time()
            print(f"  R1-7B@{MAX_BUDGET} [{done_count}/{N_SAMPLES}] acc={ok_count/done_count*100:.1f}% ceiling={ceil_count} avg_tok={avg_tok:.0f}", flush=True)

            # Checkpoint
            with open(ckpt_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "model": "DeepSeek-R1-Distill-Qwen-7B",
                        "ds": "MATH", "max_budget": MAX_BUDGET,
                        "virtual_budgets": VIRTUAL_BUDGETS,
                        "seed": SEED, "strategy": "max_budget_truncation"
                    },
                    "raw_outputs": raw_outputs
                }, f)

    print(f"\n=== Phase 2: Extract virtual budgets ===", flush=True)

    # Phase 2: For each sample, extract answers at each virtual budget
    for vb in VIRTUAL_BUDGETS:
        bkey = f"vb_{vb}"
        if bkey in all_results and all_results[bkey].get("done"):
            print(f"  @{vb} already extracted", flush=True)
            continue

        samples = []
        for i in range(N_SAMPLES):
            si = raw_outputs.get(str(i), {})
            gt = si.get("gt", "")
            gen_tok = si.get("gen_tokens", 0)
            gen_ids = si.get("gen_ids", [])

            if "err" in si:
                samples.append({"q": i, "ok": False, "tok": 0, "budget": vb,
                               "pred": "", "gt": gt, "err": si["err"], "hit_ceiling": False})
                continue

            if gen_tok <= vb:
                # Generation shorter than virtual budget — use full output
                truncated_text = tok.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
                tok_used = gen_tok
                hit_ceil = gen_tok >= vb - 5
            else:
                # Precise token-level truncation: take first vb tokens
                truncated_ids = gen_ids[:vb]
                truncated_text = tok.decode(truncated_ids, skip_special_tokens=True)
                tok_used = vb
                hit_ceil = True

            ans = extract_ans(truncated_text, "MATH")
            ok = check(ans, gt)

            samples.append({
                "q": i, "ok": ok, "tok": tok_used, "budget": vb,
                "pred": ans, "gt": gt, "hit_ceiling": hit_ceil,
            })

        all_results[bkey] = {"done": True, "samples": samples}
        acc = sum(1 for s in samples if s.get('ok')) / len(samples) * 100
        ceil = sum(1 for s in samples if s.get('hit_ceiling'))
        print(f"  @{vb}: acc={acc:.1f}% ceiling={ceil}/{len(samples)}", flush=True)

    # Save final results
    with open(ckpt_file, 'w') as f:
        json.dump({
            "metadata": {
                "model": "DeepSeek-R1-Distill-Qwen-7B",
                "ds": "MATH", "max_budget": MAX_BUDGET,
                "virtual_budgets": VIRTUAL_BUDGETS,
                "seed": SEED, "strategy": "max_budget_truncation",
                "done": True
            },
            "results": all_results,
            "raw_outputs": raw_outputs
        }, f)

    # Print summary
    print(f"\n{'='*70}")
    print(f"OVER THINKING BRIDGE: R1-Distill-Qwen-7B (Efficient Version)")
    print(f"{'='*70}")
    print(f"{'Budget':>8} {'Acc%':>8} {'CeilHit':>8} {'CeilRate':>8} {'AvgTok':>8}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    prev_acc = None
    for vb in VIRTUAL_BUDGETS:
        bkey = f"vb_{vb}"
        if bkey in all_results and all_results[bkey].get("done"):
            samples = all_results[bkey]["samples"]
            acc = sum(1 for r in samples if r.get('ok')) / len(samples) * 100
            ceil = sum(1 for r in samples if r.get('hit_ceiling'))
            ceil_rate = ceil / len(samples) * 100
            avg_tok = sum(r.get('tok',0) for r in samples) / len(samples)
            delta = f"{'+'if prev_acc and acc>=prev_acc else ''}{acc-prev_acc:.1f}%" if prev_acc else "—"
            print(f"{vb:>8} {acc:>8.1f} {ceil:>8} {ceil_rate:>7.1f}% {avg_tok:>8.0f}  Δ={delta}")
            prev_acc = acc

    # Overthinking analysis
    print(f"\n--- Overthinking Analysis ---")
    budget_accs = {}
    for vb in VIRTUAL_BUDGETS:
        bkey = f"vb_{vb}"
        if bkey in all_results:
            samples = all_results[bkey]["samples"]
            budget_accs[vb] = sum(1 for r in samples if r.get('ok')) / len(samples) * 100

    if len(budget_accs) > 2:
        peak_budget = max(budget_accs, key=budget_accs.get)
        peak_acc = budget_accs[peak_budget]
        max_budget = max(budget_accs.keys())
        max_acc = budget_accs[max_budget]

        print(f"  Peak accuracy: {peak_acc:.1f}% at @{peak_budget}")
        print(f"  @max ({max_budget}): {max_acc:.1f}%")

        if max_acc < peak_acc:
            decline = peak_acc - max_acc
            print(f"  *** OVER THINKING DETECTED: {decline:.1f}% accuracy decline ***")
        else:
            print(f"  No overthinking detected (accuracy monotonically increases)")

        # Compare with standard Qwen2.5-7B
        print(f"\n--- Comparison: R1 vs Standard Qwen2.5-7B ---")
        print(f"  Standard 7B: @256=53.5%, @512=78.5%, @1024=80.5% (NO overthinking)")
        if 256 in budget_accs:
            print(f"  R1-Distill-7B: @256={budget_accs[256]:.1f}% (truncation dominated)")

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
