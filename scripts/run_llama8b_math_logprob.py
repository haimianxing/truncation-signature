#!/usr/bin/env python3
"""
Collect logprob for LLaMA-3-8B-Instruct on MATH @256.
Fills the gap in the multi-model ROC figure.

Usage: CUDA_VISIBLE_DEVICES=2 python3 -u run_llama8b_math_logprob.py
"""
import json, re, gc
from pathlib import Path
from datetime import datetime
import torch, numpy as np

BASE = Path(__file__).parent
CKPT = BASE / "results_v2" / "logprob_collection" / "llama3_8b_math_256.json"
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


def main():
    print(f"LLaMA-3-8B MATH Logprob | {datetime.now()}", flush=True)

    if CKPT.exists():
        print(f"Already cached: {CKPT}", flush=True)
        with open(CKPT) as f:
            results = json.load(f)
        labels = np.array([1 if r['ok'] else 0 for r in results])
        logprobs = np.array([r['mean_logprob'] for r in results])
        from sklearn.metrics import roc_auc_score
        print(f"Logprob AUC: {roc_auc_score(labels, logprobs):.3f}")
        return

    device = "cuda:0"
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/mnt/data/pre_model/Llama-3-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    with open(BASE / "math_real_200.json") as f:
        problems = json.load(f)

    results = []
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

        results.append({
            "q": i, "ok": ok, "tok": gen_tokens,
            "mean_logprob": mean_logprob,
        })
        if (i + 1) % 50 == 0:
            acc = sum(x['ok'] for x in results) / len(results) * 100
            print(f"  [{i+1}/200] acc={acc:.1f}%", flush=True)

    with open(CKPT, 'w') as f:
        json.dump(results, f, indent=2)

    labels = np.array([1 if r['ok'] else 0 for r in results])
    logprobs = np.array([r['mean_logprob'] for r in results])
    tokens = np.array([r['tok'] for r in results])
    from sklearn.metrics import roc_auc_score
    print(f"\nToken AUC: {roc_auc_score(1-labels, tokens):.3f}")
    print(f"Logprob AUC: {roc_auc_score(labels, logprobs):.3f}")
    print(f"Saved: {CKPT}")
    print(f"Done: {datetime.now()}", flush=True)

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
