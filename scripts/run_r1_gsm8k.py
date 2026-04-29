#!/usr/bin/env python3
"""
R1-Distill-7B GSM8K Multi-Budget Experiment
=============================================
Generate at @8192, extract virtual budgets via truncation.
Greedy decoding ensures equivalence.
Usage: CUDA_VISIBLE_DEVICES=3 python3 -u run_r1_gsm8k.py
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np

SEED = 42
N_SAMPLES = 200
MAX_BUDGET = 8192
BASE = Path(__file__).parent
DEVICE = "cuda:0"

VIRTUAL_BUDGETS = [256, 512, 1024, 2048, 4096, 8192]

MODEL_PATH = "/mnt/data/pre_model/DeepSeek-R1-Distill-Qwen-7B"
DATA_FILE = BASE / "gsm8k_real_200.json"
CKPT_DIR = BASE / "results_v2" / "r1_gsm8k"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def extract_ans(text):
    # GSM8K: look for #### or final number
    if "####" in text:
        nums = re.findall(r'-?\d+\.?\d*', text.split("####")[-1])
        if nums: return nums[-1]
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

def main():
    print(f"R1-Distill-7B GSM8K Multi-Budget | {datetime.now()}", flush=True)
    print(f"Device: {DEVICE}, Strategy: @{MAX_BUDGET} → virtual budget extraction", flush=True)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found!", flush=True)
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    with open(DATA_FILE) as f:
        alldata = json.load(f)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = alldata[:N_SAMPLES]

    ckpt_file = CKPT_DIR / "r1_gsm8k_8192.json"
    raw_outputs = {}
    all_results = {}

    if ckpt_file.exists():
        with open(ckpt_file) as f:
            saved = json.load(f)
        raw_outputs = saved.get("raw_outputs", {})
        all_results = saved.get("results", {})
        print(f"Resuming: {len(raw_outputs)} samples done", flush=True)

    # Phase 1: Generate all @8192
    start_idx = len(raw_outputs)
    print(f"\n=== Phase 1: Generate {N_SAMPLES} samples @{MAX_BUDGET} (from idx {start_idx}) ===", flush=True)

    for i in range(start_idx, N_SAMPLES):
        q = data[i]["query"]
        gt = str(data[i].get("ground_truth", data[i].get("answer", "")))

        try:
            content = q + "\nLet's think step by step."
            messages = [{"role": "user", "content": content}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=MAX_BUDGET,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            lat = time.time() - t0

            gen_tok = out.shape[1] - inp["input_ids"].shape[1]
            gen_ids = out[0][inp["input_ids"].shape[1]:].tolist()
            full_text = tok.decode(gen_ids, skip_special_tokens=True)

            raw_outputs[str(i)] = {
                "q": i, "gt": gt,
                "gen_tokens": gen_tok,
                "hit_ceiling": gen_tok >= MAX_BUDGET - 5,
                "full_ans": extract_ans(full_text),
                "gen_ids": gen_ids,
            }
        except Exception as e:
            raw_outputs[str(i)] = {"q": i, "gt": gt, "err": str(e), "gen_tokens": 0}

        done = len(raw_outputs)
        if done % 10 == 0 or done == N_SAMPLES:
            avg_tok = sum(v.get("gen_tokens", 0) for v in raw_outputs.values()) / len(raw_outputs)
            print(f"  [{done}/{N_SAMPLES}] avg_tok={avg_tok:.0f}", flush=True)
            with open(ckpt_file, 'w') as f:
                json.dump({"raw_outputs": raw_outputs, "results": all_results}, f)

    # Phase 2: Extract virtual budgets
    print(f"\n=== Phase 2: Extract virtual budgets ===", flush=True)

    for vb in VIRTUAL_BUDGETS:
        bkey = f"vb_{vb}"
        ckpt_vb = CKPT_DIR / f"r1_gsm8k_{vb}.json"

        if ckpt_vb.exists():
            print(f"  @{vb} already cached", flush=True)
            with open(ckpt_vb) as f:
                all_results[bkey] = json.load(f)
            continue

        samples = []
        for i in range(N_SAMPLES):
            si = raw_outputs.get(str(i), {})
            gt = si.get("gt", "")
            gen_tok = si.get("gen_tokens", 0)
            gen_ids = si.get("gen_ids", [])

            if "err" in si:
                samples.append({"q": i, "ok": False, "tok": 0, "truncated": False})
                continue

            if gen_tok <= vb:
                truncated_text = tok.decode(gen_ids, skip_special_tokens=True)
                tok_used = gen_tok
                truncated = gen_tok >= vb - 5
            else:
                truncated_ids = gen_ids[:vb]
                truncated_text = tok.decode(truncated_ids, skip_special_tokens=True)
                tok_used = vb
                truncated = True

            ans = extract_ans(truncated_text)
            ok = check(ans, gt)
            samples.append({"q": i, "ok": ok, "ans": ans, "gen_tok": tok_used, "truncated": truncated})

        acc = sum(s["ok"] for s in samples) / len(samples) * 100
        trunc_rate = sum(s["truncated"] for s in samples) / len(samples) * 100
        result = {"results": samples, "accuracy": acc, "truncation_rate": trunc_rate,
                  "condition": f"base_{vb}", "dataset": "GSM8K", "model": "R1-Distill-7B",
                  "budget": vb, "n_questions": len(samples)}
        all_results[bkey] = result

        with open(ckpt_vb, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  @{vb}: acc={acc:.1f}% trunc={trunc_rate:.0f}%", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"R1-Distill-7B GSM8K Budget Spectrum", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Budget':>8} {'Acc':>8} {'CeilRate':>10} {'AvgTok':>8}", flush=True)
    for vb in VIRTUAL_BUDGETS:
        bkey = f"vb_{vb}"
        if bkey in all_results:
            r = all_results[bkey]
            results = r.get("results", r.get("samples", []))
            acc = sum(s["ok"] for s in results) / len(results) * 100
            trunc = sum(s["truncated"] for s in results) / len(results) * 100
            avg_tok = sum(s.get("gen_tok", s.get("tok", 0)) for s in results) / len(results)
            print(f"{vb:>8} {acc:>7.1f}% {trunc:>9.1f}% {avg_tok:>8.0f}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
