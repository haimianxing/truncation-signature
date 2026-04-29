#!/usr/bin/env python3
"""
Efficiency Frontier — Qwen2.5-7B MATH @128/256/384/512/768/1024
6-point curve for the larger model to complement 3B curve.
Cross-model efficiency frontier comparison = unique contribution.
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
N_SAMPLES = 200
BASE = Path(__file__).parent

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

def make_input(tok, question):
    content = f"{question}\n\nPlease think step by step, then give your final numerical answer after ####."
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")

def main():
    device = "cuda:0"
    print(f"Efficiency Frontier 7B | {datetime.now()}", flush=True)

    model_path = "/mnt/data/pre_model/Qwen2.5-7B-Instruct"
    data_file = BASE / "math_real_200.json"
    ckpt_dir = BASE / "results_frontier"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Loading Qwen2.5-7B from {model_path}", flush=True)
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

    budgets = [128, 256, 384, 512, 768, 1024]
    all_results = {}

    ckpt_file = ckpt_dir / "frontier_7b_math.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            saved = json.load(f)
        all_results = saved.get("results", {})

    for budget in budgets:
        bkey = str(budget)
        if bkey in all_results and all_results[bkey].get("done"):
            acc = sum(1 for r in all_results[bkey]["samples"] if r.get('ok')) / len(all_results[bkey]["samples"]) * 100
            print(f"  @budget={budget} already done: acc={acc:.1f}%", flush=True)
            continue

        print(f"\n--- Budget={budget} ---", flush=True)
        samples = []
        for i in range(N_SAMPLES):
            q = data[i]["query"]
            gt = str(data[i].get("ground_truth", ""))

            try:
                inp = make_input(tok, q)
                inp = inp.to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=budget,
                        do_sample=False,
                        temperature=None,
                        pad_token_id=tok.eos_token_id,
                    )

                gen_tokens = out.shape[1] - inp["input_ids"].shape[1]
                response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                ans = extract_ans(response, "MATH")
                ok = check(ans, gt)

                samples.append({
                    "q": i, "ok": ok, "tok": gen_tokens, "budget": budget,
                    "pred": ans, "gt": gt,
                    "hit_ceiling": gen_tokens >= budget - 5,
                })
            except Exception as e:
                samples.append({
                    "q": i, "ok": False, "tok": 0, "budget": budget,
                    "pred": "", "gt": gt, "err": str(e), "hit_ceiling": False,
                })

            if (i+1) % 50 == 0 or (i+1) == N_SAMPLES:
                acc = sum(1 for r in samples if r.get('ok')) / len(samples) * 100
                ceiling_hits = sum(1 for r in samples if r.get('hit_ceiling'))
                print(f"  Qwen2.5-7B@{budget} [{i+1}/{N_SAMPLES}] acc={acc:.1f}% ceiling_hits={ceiling_hits}", flush=True)

                # Checkpoint
                all_results[bkey] = {"done": (i+1)==N_SAMPLES, "samples": samples}
                with open(ckpt_file, 'w') as f:
                    json.dump({"metadata": {"model": "Qwen2.5-7B", "ds": "MATH",
                                           "budgets": budgets, "seed": SEED},
                               "results": all_results}, f)

        all_results[bkey] = {"done": True, "samples": samples}

    # Save final
    with open(ckpt_file, 'w') as f:
        json.dump({"metadata": {"model": "Qwen2.5-7B", "ds": "MATH",
                               "budgets": budgets, "seed": SEED, "done": True},
                   "results": all_results}, f)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EFFICIENCY FRONTIER: Qwen2.5-7B MATH")
    print(f"{'='*60}")
    print(f"{'Budget':>8} {'Acc%':>8} {'CeilHit':>8} {'CeilRate':>8}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for budget in budgets:
        bkey = str(budget)
        if bkey in all_results and all_results[bkey].get("done"):
            samples = all_results[bkey]["samples"]
            acc = sum(1 for r in samples if r.get('ok')) / len(samples) * 100
            ceil = sum(1 for r in samples if r.get('hit_ceiling'))
            ceil_rate = ceil / len(samples) * 100
            print(f"{budget:>8} {acc:>8.1f} {ceil:>8} {ceil_rate:>7.1f}%")

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
