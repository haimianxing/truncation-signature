#!/usr/bin/env python3
"""
MULTI-SEED VALIDATION for Ratchet + Token AUC
Validates key findings across seeds 123 and 456.
Critical conditions: 7B MATH, 3B MATH (highest ratios)
"""
import sys, os, json, time, re, gc, random
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr

BASE = Path(__file__).parent
N_SAMPLES = 200
SEEDS = [123, 456]

MODELS_METHODS = [
    # (model_name, model_path, dataset, methods)
    ("Qwen2.5-7B", "/mnt/data/pre_model/Qwen2.5-7B-Instruct", "MATH",
     [("cot_t0_256", True, 256), ("cot_t0_512", True, 512)]),
    ("Qwen2.5-3B", "/mnt/data/pre_model/Qwen2.5-3B-Instruct", "MATH",
     [("cot_t0_256", True, 256), ("cot_t0_512", True, 512)]),
]

def extract_ans(text):
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
            nums = re.findall(r'-?\d+\.?\d*', m.group(1) if 'answer' in pat.lower() else m.group(0))
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
    print(f"Multi-seed validation | {datetime.now()}", flush=True)

    for model_name, model_path, dataset, methods in MODELS_METHODS:
        data_file = BASE / f"{dataset.lower()}_real_200.json"
        with open(data_file) as f:
            alldata = json.load(f)

        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        for seed in SEEDS:
            for method_name, use_cot, max_tok in methods:
                ckpt_dir = BASE / "results_math_v2"
                ckpt_file = ckpt_dir / f"{model_name}_{dataset}_{method_name}_s{seed}.json"

                if ckpt_file.exists():
                    with open(ckpt_file) as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("done"):
                        acc = sum(1 for r in existing["results"] if r.get("ok"))/len(existing["results"])*100
                        print(f"  SKIP {model_name}/{method_name}/s{seed} (acc={acc:.1f}%)", flush=True)
                        continue
                    results = existing.get("results", [])
                    start = len(results)
                else:
                    results = []
                    start = 0

                if start >= N_SAMPLES:
                    continue

                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                data = alldata[:N_SAMPLES]

                for i in range(start, N_SAMPLES):
                    q = data[i]["query"]
                    gt = str(data[i].get("ground_truth", ""))
                    try:
                        inp = make_input(tok, q).to(device)
                        with torch.no_grad():
                            out = model.generate(**inp, max_new_tokens=max_tok,
                                               do_sample=False, pad_token_id=tok.eos_token_id)
                        response = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                        tokn = out.shape[1] - inp["input_ids"].shape[1]
                        ans = extract_ans(response)
                        ok = check(ans, gt)
                        results.append({"q": i, "ok": ok, "tok": tokn, "pred": ans, "gt": gt})
                    except Exception as e:
                        results.append({"q": i, "ok": False, "tok": 0, "pred": "", "gt": gt, "err": str(e)})

                    if (i+1) % 50 == 0 or (i+1) == N_SAMPLES:
                        done = (i+1) == N_SAMPLES
                        with open(ckpt_file, 'w') as f:
                            json.dump({"metadata": {"model": model_name, "ds": dataset,
                                                   "method": method_name, "seed": seed,
                                                   "done": done, "n": i+1}, "results": results}, f)
                        acc = sum(1 for r in results if r.get('ok'))/len(results)*100
                        print(f"  {model_name}/{method_name}/s{seed} [{i+1}/{N_SAMPLES}] acc={acc:.1f}%", flush=True)

        del model; torch.cuda.empty_cache(); gc.collect()

    # Cross-seed analysis
    print(f"\n{'='*60}")
    print("CROSS-SEED RATCHET VALIDATION")
    print(f"{'='*60}")

    for model_name, _, dataset, methods in MODELS_METHODS:
        for seed in [42] + SEEDS:
            f256 = BASE/"results_math_v2"/f"{model_name}_{dataset}_cot_t0_256_s{seed}.json"
            f512 = BASE/"results_math_v2"/f"{model_name}_{dataset}_cot_t0_512_s{seed}.json"
            if not f256.exists() or not f512.exists():
                continue
            with open(f256) as f: d256 = json.load(f)['results']
            with open(f512) as f: d512 = json.load(f)['results']
            if not (json.load(open(f256)).get("metadata",{}).get("done") and json.load(open(f512)).get("metadata",{}).get("done")):
                # re-open since we consumed it
                with open(f256) as f: d256 = json.load(f)['results']
                with open(f512) as f: d512 = json.load(f)['results']
            acc256 = sum(1 for r in d256 if r['ok'])/len(d256)*100
            acc512 = sum(1 for r in d512 if r['ok'])/len(d512)*100
            recovery = sum(1 for a,b in zip(d256, d512) if not a['ok'] and b['ok'])
            collapse = sum(1 for a,b in zip(d256, d512) if a['ok'] and not b['ok'])
            wrong = sum(1 for r in d256 if not r['ok'])
            ratio = f"{recovery}:{collapse}" if collapse > 0 else f"{recovery}:0"
            
            labels = [0 if r['ok'] else 1 for r in d256]
            tokens = [r['tok'] for r in d256]
            auc = roc_auc_score(labels, tokens) if len(set(labels))>1 else 0
            
            print(f"  {model_name:18s} s{seed:3d} | @256={acc256:.1f}% @512={acc512:.1f}% | "
                  f"R={recovery}/{wrong} C={collapse} ({ratio}) | AUC={auc:.3f}")

    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
