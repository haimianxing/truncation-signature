#!/usr/bin/env python3
"""
Collect logprob AUC for Gemma-2-9B-IT (Table 2 gaps).
Conditions:
  1. Gemma-2-9B-IT MATH @256
  2. Gemma-2-9B-IT GSM8K @256

Usage: CUDA_VISIBLE_DEVICES=1 python3 -u run_gemma9b_logprob.py
"""
import sys, os, json, re, gc
from pathlib import Path
from datetime import datetime
import torch, numpy as np

BASE = Path(__file__).parent
CKPT_DIR = BASE / "results_v2" / "logprob_collection"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BUDGET = 256

def extract_answer_math(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).strip()
    m = re.search(r'(?:the answer is|therefore)[\s]*([-$\d\.,/]+)', text, re.I)
    if m: return m.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()

def check_answer(pred, gold):
    pred = pred.strip().replace(',', '').replace('$', '')
    gold = gold.strip().replace(',', '').replace('$', '')
    if pred == gold: return True
    try: return abs(float(pred) - float(gold)) < 1e-6
    except: return pred == gold

def compute_auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)

def main():
    print(f"Gemma-2-9B Logprob Collection | {datetime.now()}", flush=True)
    device = "cuda:0"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.metrics import roc_auc_score

    model_path = "/mnt/data2/zcz/gemma-2-9b-it/google/gemma-2-9b-it"

    # ================================================================
    # Task 1: Gemma-2-9B-IT MATH @256
    # ================================================================
    ckpt1 = CKPT_DIR / "gemma2_9b_math_256.json"
    if ckpt1.exists():
        print(f"Loading cached: {ckpt1}", flush=True)
        with open(ckpt1) as f:
            r1 = json.load(f)
    else:
        print(f"\n=== Gemma-2-9B-IT MATH @256 ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        with open(BASE / "math_real_200.json") as f:
            problems = json.load(f)

        r1 = []
        for i, prob in enumerate(problems):
            messages = [{"role": "user", "content": prob['query']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=BUDGET,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                )
            gen_ids = out.sequences[0][inp["input_ids"].shape[1]:].tolist()
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            gen_tokens = len(gen_ids)

            # Compute mean logprob
            if len(out.scores) > 0:
                log_probs = []
                for t, score in enumerate(out.scores[:gen_tokens]):
                    if t < len(gen_ids):
                        log_prob = torch.log_softmax(score[0], dim=-1)[gen_ids[t]].item()
                        log_probs.append(log_prob)
                mean_logprob = np.mean(log_probs) if log_probs else 0.0
            else:
                mean_logprob = 0.0

            pred = extract_answer_math(gen_text)
            ok = check_answer(pred, prob['ground_truth'])
            ceiling = gen_tokens >= BUDGET - 5

            r1.append({
                "q": i, "ok": ok, "tok": gen_tokens,
                "mean_logprob": mean_logprob,
                "ceiling_hit": ceiling,
            })
            if (i+1) % 50 == 0:
                acc = sum(x['ok'] for x in r1)/len(r1)*100
                print(f"  [{i+1}/200] acc={acc:.1f}%", flush=True)

        with open(ckpt1, 'w') as f:
            json.dump(r1, f, indent=2)

        # Compute AUC
        labels = np.array([1 if x['ok'] else 0 for x in r1])
        token_counts = np.array([x['tok'] for x in r1])
        logprobs = np.array([x['mean_logprob'] for x in r1])

        token_auc = compute_auc(labels, token_counts)
        logprob_auc = compute_auc(labels, logprobs)
        print(f"  Token AUC: {token_auc:.3f}" if token_auc else "  Token AUC: N/A")
        print(f"  Logprob AUC: {logprob_auc:.3f}" if logprob_auc else "  Logprob AUC: N/A")

        del model; torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # Task 2: Gemma-2-9B-IT GSM8K @256
    # ================================================================
    ckpt2 = CKPT_DIR / "gemma2_9b_gsm8k_256.json"
    if ckpt2.exists():
        print(f"Loading cached: {ckpt2}", flush=True)
        with open(ckpt2) as f:
            r2 = json.load(f)
    else:
        print(f"\n=== Gemma-2-9B-IT GSM8K @256 ===", flush=True)
        # Reload model if needed
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        with open(BASE / "gsm8k_real_200.json") as f:
            problems = json.load(f)

        r2 = []
        for i, prob in enumerate(problems):
            messages = [{"role": "user", "content": prob['query']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=BUDGET,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                )
            gen_ids = out.sequences[0][inp["input_ids"].shape[1]:].tolist()
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            gen_tokens = len(gen_ids)

            # Mean logprob
            if len(out.scores) > 0:
                log_probs = []
                for t, score in enumerate(out.scores[:gen_tokens]):
                    if t < len(gen_ids):
                        log_prob = torch.log_softmax(score[0], dim=-1)[gen_ids[t]].item()
                        log_probs.append(log_prob)
                mean_logprob = np.mean(log_probs) if log_probs else 0.0
            else:
                mean_logprob = 0.0

            pred = extract_answer_math(gen_text)
            ok = check_answer(pred, prob['ground_truth'])
            ceiling = gen_tokens >= BUDGET - 5

            r2.append({
                "q": i, "ok": ok, "tok": gen_tokens,
                "mean_logprob": mean_logprob,
                "ceiling_hit": ceiling,
            })
            if (i+1) % 50 == 0:
                acc = sum(x['ok'] for x in r2)/len(r2)*100
                print(f"  [{i+1}/200] acc={acc:.1f}%", flush=True)

        with open(ckpt2, 'w') as f:
            json.dump(r2, f, indent=2)

        # Compute AUC
        labels = np.array([1 if x['ok'] else 0 for x in r2])
        token_counts = np.array([x['tok'] for x in r2])
        logprobs = np.array([x['mean_logprob'] for x in r2])

        token_auc = compute_auc(labels, token_counts)
        logprob_auc = compute_auc(labels, logprobs)
        print(f"  Token AUC: {token_auc:.3f}" if token_auc else "  Token AUC: N/A")
        print(f"  Logprob AUC: {logprob_auc:.3f}" if logprob_auc else "  Logprob AUC: N/A")

        del model; torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    # Gemma-2-9B MATH
    labels = np.array([1 if x['ok'] else 0 for x in r1])
    tc = np.array([x['tok'] for x in r1])
    lp = np.array([x['mean_logprob'] for x in r1])
    t_auc = compute_auc(labels, tc)
    l_auc = compute_auc(labels, lp)
    print(f"Gemma-2-9B MATH @256:", flush=True)
    print(f"  Token AUC: {t_auc:.3f}" if t_auc else "  Token AUC: N/A")
    print(f"  Logprob AUC: {l_auc:.3f}" if l_auc else "  Logprob AUC: N/A")

    # Gemma-2-9B GSM8K
    labels = np.array([1 if x['ok'] else 0 for x in r2])
    tc = np.array([x['tok'] for x in r2])
    lp = np.array([x['mean_logprob'] for x in r2])
    t_auc = compute_auc(labels, tc)
    l_auc = compute_auc(labels, lp)
    print(f"Gemma-2-9B GSM8K @256:", flush=True)
    print(f"  Token AUC: {t_auc:.3f}" if t_auc else "  Token AUC: N/A")
    print(f"  Logprob AUC: {l_auc:.3f}" if l_auc else "  Logprob AUC: N/A")

    print(f"\nDone: {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
