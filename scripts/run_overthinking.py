#!/usr/bin/env python3
"""
Overthinking Experiment: DeepSeek-R1-Distill + Scale-Dependent Analysis
========================================================================
Purpose: Extend TTS-Bench findings to reasoning-specialized models (DeepSeek-R1)
and systematically connect to overthinking research.

Key hypotheses to test:
  H1: DeepSeek-R1-Distill-Qwen-7B shows STRONGER CoT penalty than
      Qwen2.5-7B-Instruct (prediction from "Reasoning Autonomy Transition")
  H2: DeepSeek-R1 produces very long thinking tokens that get truncated at
      256 budget → Truncation Paradox is even more severe
  H3: The answer distribution from DeepSeek-R1 is even MORE systematic
      (lower entropy) → BoN should hurt even more
  H4: Budget forcing (limiting thinking tokens) creates a U-shaped curve
      for reasoning models — too little = truncated, too much = overthinking

Models: DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct (comparison)
Dataset: MATH (50 questions)
Budgets: 256, 512, 1024 tokens
Device: cuda:2 (free GPU)
"""
import sys, os, json, time, gc, random, re, warnings
import torch, numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
SEED = 42
N_QUESTIONS = 50
N_BON = 8
DEVICE = "cuda:2"
BUDGETS = [256, 512, 1024]

MODELS = {
    "DeepSeek-R1-Distill-7B": "/mnt/data/pre_model/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct": "/mnt/data/pre_model/Qwen2.5-7B-Instruct",
}

DATA_FILE = BASE / "math_real_200.json"
CKPT_DIR = BASE / "results_v2" / "overthinking"


def extract_ans(text):
    # Remove thinking tags for DeepSeek-R1
    text_clean = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', text, flags=re.DOTALL)
    for txt in [text_clean, text]:
        boxed = re.findall(r'\\boxed\{([^}]+)\}', txt)
        if boxed:
            nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
            if nums: return nums[-1]
        for pat in [
            r'(?:the answer is|therefore[,:\s]+|thus[,:\s]+|so the answer is|final answer[:\s]+)([^\n.]+)',
            r'answer[:\s]+([^\n.]+)',
        ]:
            matches = list(re.finditer(pat, txt, re.IGNORECASE))
            if matches:
                nums = re.findall(r'-?\d+\.?\d*', matches[-1].group(1))
                if nums: return nums[-1]
        lines = txt.strip().split('\n')
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


def parse_thinking(text):
    """Parse DeepSeek-R1 thinking tokens."""
    think_match = re.search(r'<think[^>]*>(.*?)</think[^>]*>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1)
        answer_part = text[think_match.end():]
        return thinking, answer_part
    return "", text


def run_experiment(model, tok, questions, model_name, budget, prompt_type="baseline",
                   use_chat=True, do_sample=False, temperature=0.7):
    """Run inference at a given budget and prompt type."""
    model.eval()
    results = []

    for i, q_data in enumerate(questions):
        q = q_data["query"]
        gt = str(q_data.get("ground_truth", q_data.get("answer", "")))

        if prompt_type == "cot":
            content = q + "\nLet's think step by step."
        else:
            content = q

        if use_chat:
            messages = [{"role": "user", "content": content}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = content

        inp = tok(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": budget,
            "pad_token_id": tok.eos_token_id,
        }
        if do_sample:
            random.seed(SEED); np.random.seed(SEED)
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
        else:
            gen_kwargs["do_sample"] = False

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inp, **gen_kwargs)
        lat = time.time() - t0

        gen_text = tok.decode(out[0], skip_special_tokens=False)
        full_text = tok.decode(out[0], skip_special_tokens=True)

        # Parse thinking vs answer for DeepSeek-R1
        thinking_text, answer_text = parse_thinking(gen_text)
        thinking_tokens = len(tok.encode(thinking_text))
        answer_tokens = len(tok.encode(answer_text))
        gen_tok = out.shape[1] - inp["input_ids"].shape[1]
        truncated = gen_tok >= budget - 5

        ans = extract_ans(full_text)

        results.append({
            "q": i, "ok": check(ans, gt), "ans": ans,
            "gen_tok": gen_tok, "think_tok": thinking_tokens,
            "answer_tok": answer_tokens, "lat": round(lat, 4),
            "truncated": truncated, "has_think_tag": bool(thinking_text),
        })

        if (i+1) % 10 == 0:
            acc = sum(r["ok"] for r in results) / len(results) * 100
            trunc = sum(r["truncated"] for r in results) / len(results) * 100
            print(f"    [{model_name}/{prompt_type}/{budget}tok] {i+1}/{len(questions)} "
                  f"acc={acc:.1f}% trunc={trunc:.0f}%", flush=True)

    return results


def compute_answer_entropy(model, tok, questions, model_name, n_samples=8, use_chat=True):
    """BoN answer distribution entropy."""
    model.eval()
    entropies = []
    consensus_rates = []

    for i, q_data in enumerate(questions):
        q = q_data["query"]
        if use_chat:
            messages = [{"role": "user", "content": q}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = q
        inp = tok(prompt, return_tensors="pt").to(model.device)

        answers = []
        for s in range(n_samples):
            random.seed(SEED + s); np.random.seed(SEED + s)
            torch.manual_seed(SEED + s); torch.cuda.manual_seed_all(SEED + s)

            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0], skip_special_tokens=True)
            if prompt in gen: gen = gen[len(prompt):]
            answers.append(extract_ans(gen))

        counts = Counter(answers)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        entropies.append(entropy)
        consensus_rates.append(max(counts.values()) / total)

        if (i+1) % 10 == 0:
            print(f"    [entropy/{model_name}] {i+1}/{len(questions)} H={np.mean(entropies):.3f}", flush=True)

    return {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "mean_consensus": float(np.mean(consensus_rates)),
    }


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Overthinking Experiment | {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    with open(DATA_FILE) as f:
        data = json.load(f)
    questions = data[:N_QUESTIONS]
    print(f"Using {len(questions)} MATH questions, budgets={BUDGETS}", flush=True)

    all_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*70}", flush=True)

        is_reasoning = "DeepSeek" in model_name or "R1" in model_name

        print(f"Loading {model_name}...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        print(f"  Loaded: {model.config.num_hidden_layers} layers", flush=True)

        model_results = {}

        # === Budget sweep (baseline, no CoT) ===
        for budget in BUDGETS:
            key = f"baseline_{budget}"
            ckpt = CKPT_DIR / f"{model_name}_{key}.json"
            if ckpt.exists():
                print(f"  SKIP {key} (cached)", flush=True)
                model_results[key] = json.load(open(ckpt))
                continue

            print(f"\n  Running {key}...", flush=True)
            results = run_experiment(model, tok, questions, model_name, budget,
                                     "baseline", use_chat=True, do_sample=False)
            acc = sum(r["ok"] for r in results) / len(results) * 100
            trunc = sum(r["truncated"] for r in results) / len(results) * 100
            think_toks = np.mean([r["think_tok"] for r in results])
            ans_toks = np.mean([r["answer_tok"] for r in results])

            save = {"results": results, "accuracy": acc, "truncation_rate": trunc,
                    "mean_think_tok": float(think_toks), "mean_answer_tok": float(ans_toks),
                    "budget": budget, "prompt_type": "baseline"}
            with open(ckpt, 'w') as f:
                json.dump(save, f, indent=2)
            model_results[key] = save
            print(f"  => {key}: acc={acc:.1f}%, trunc={trunc:.0f}%, "
                  f"think_tok={think_toks:.0f}, ans_tok={ans_toks:.0f}", flush=True)

        # === CoT at 512 ===
        key = "cot_512"
        ckpt = CKPT_DIR / f"{model_name}_{key}.json"
        if ckpt.exists():
            print(f"  SKIP {key} (cached)", flush=True)
            model_results[key] = json.load(open(ckpt))
        else:
            print(f"\n  Running {key}...", flush=True)
            results = run_experiment(model, tok, questions, model_name, 512,
                                     "cot", use_chat=True, do_sample=False)
            acc = sum(r["ok"] for r in results) / len(results) * 100
            save = {"results": results, "accuracy": acc, "budget": 512, "prompt_type": "cot"}
            with open(ckpt, 'w') as f:
                json.dump(save, f, indent=2)
            model_results[key] = save
            print(f"  => {key}: acc={acc:.1f}%", flush=True)

        # === Answer Entropy ===
        ent_key = "answer_entropy"
        ent_ckpt = CKPT_DIR / f"{model_name}_{ent_key}.json"
        if ent_ckpt.exists():
            model_results[ent_key] = json.load(open(ent_ckpt))
        else:
            print(f"\n  Computing answer entropy...", flush=True)
            ent = compute_answer_entropy(model, tok, questions, model_name, N_BON)
            with open(ent_ckpt, 'w') as f:
                json.dump(ent, f, indent=2)
            model_results[ent_key] = ent
            print(f"  => entropy={ent['mean_entropy']:.3f}, consensus={ent['mean_consensus']:.3f}", flush=True)

        all_results[model_name] = model_results
        del model; torch.cuda.empty_cache(); gc.collect()
        print(f"\n  {model_name} done", flush=True)

    # === Summary ===
    print(f"\n{'='*80}", flush=True)
    print("OVERTHINKING EXPERIMENT SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<28} {'B256':>6} {'B512':>6} {'B1024':>6} {'CoT512':>7} "
          f"{'CoTΔ':>6} {'Entr':>6} {'Cons':>6}", flush=True)
    print("-" * 76, flush=True)

    for name, mr in all_results.items():
        b256 = mr.get("baseline_256", {}).get("accuracy", float('nan'))
        b512 = mr.get("baseline_512", {}).get("accuracy", float('nan'))
        b1024 = mr.get("baseline_1024", {}).get("accuracy", float('nan'))
        cot512 = mr.get("cot_512", {}).get("accuracy", float('nan'))
        cot_delta = cot512 - b512 if not np.isnan(cot512) and not np.isnan(b512) else float('nan')
        ent = mr.get("answer_entropy", {}).get("mean_entropy", float('nan'))
        con = mr.get("answer_entropy", {}).get("mean_consensus", float('nan'))
        print(f"{name:<28} {b256:>6.1f} {b512:>6.1f} {b1024:>6.1f} {cot512:>7.1f} "
              f"{cot_delta:>+6.1f} {ent:>6.3f} {con:>6.3f}", flush=True)

    # === Hypothesis Verification ===
    print(f"\n{'='*80}", flush=True)
    print("HYPOTHESIS VERIFICATION", flush=True)
    print(f"{'='*80}", flush=True)

    r1 = all_results.get("DeepSeek-R1-Distill-7B", {})
    q7 = all_results.get("Qwen2.5-7B-Instruct", {})

    # H1: R1 shows stronger CoT penalty
    r1_cot_delta = r1.get("cot_512", {}).get("accuracy", 0) - r1.get("baseline_512", {}).get("accuracy", 0)
    q7_cot_delta = q7.get("cot_512", {}).get("accuracy", 0) - q7.get("baseline_512", {}).get("accuracy", 0)
    print(f"\n[H1] CoT Penalty Comparison (prediction: R1 > Qwen2.5):", flush=True)
    print(f"  DeepSeek-R1-Distill-7B: CoT Δ = {r1_cot_delta:+.1f}%", flush=True)
    print(f"  Qwen2.5-7B-Instruct:    CoT Δ = {q7_cot_delta:+.1f}%", flush=True)
    r1_stronger = abs(r1_cot_delta) > abs(q7_cot_delta) if r1_cot_delta < 0 else False
    print(f"  => R1 has STRONGER CoT penalty: {r1_stronger}", flush=True)

    # H2: R1 has more severe truncation
    r1_trunc = r1.get("baseline_256", {}).get("truncation_rate", float('nan'))
    q7_trunc = q7.get("baseline_256", {}).get("truncation_rate", float('nan'))
    print(f"\n[H2] Truncation at 256 tokens:", flush=True)
    print(f"  R1-Distill-7B: {r1_trunc:.0f}%", flush=True)
    print(f"  Qwen2.5-7B:    {q7_trunc:.0f}%", flush=True)

    # H3: R1 has lower entropy (more systematic errors)
    r1_ent = r1.get("answer_entropy", {}).get("mean_entropy", float('nan'))
    q7_ent = q7.get("answer_entropy", {}).get("mean_entropy", float('nan'))
    print(f"\n[H3] Answer Entropy:", flush=True)
    print(f"  R1-Distill-7B: {r1_ent:.3f}", flush=True)
    print(f"  Qwen2.5-7B:    {q7_ent:.3f}", flush=True)

    # H4: Budget curve
    print(f"\n[H4] Budget-Accuracy Curve:", flush=True)
    for name, mr in all_results.items():
        curve = []
        for b in BUDGETS:
            acc = mr.get(f"baseline_{b}", {}).get("accuracy", float('nan'))
            curve.append(f"{b}:{acc:.1f}%")
        print(f"  {name}: {' → '.join(curve)}", flush=True)

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
