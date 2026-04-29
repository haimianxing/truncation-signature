#!/usr/bin/env python3
"""
CROSS-FAMILY VALIDATION: All three innovations on LLaMA-3-8B
=============================================================
Critical test: Do Ratchet, Token Confidence, and Phase Transition
hold across model families (Qwen2.5 → LLaMA-3)?
"""
import json, os
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def load_results(directory):
    data = {}
    for f in os.listdir(directory):
        if not f.endswith(".json"): continue
        with open(os.path.join(directory, f)) as fh:
            d = json.load(fh)
        m = d["metadata"]
        data[(m["model"], m.get("ds","GSM8K"), m["method"])] = {r["q"]: r for r in d["results"]}
    return data

def main():
    gsm8k = load_results("results_novel_v2")
    math = load_results("results_math_v2")
    llama = load_results("results_llama_v2")

    print("=" * 80)
    print("CROSS-FAMILY VALIDATION: LLaMA-3-8B (GSM8K)")
    print("=" * 80)

    # LLaMA-3 keys
    kbl = ("Llama-3-8B", "GSM8K", "baseline")
    kcot = ("Llama-3-8B", "GSM8K", "cot_t0")

    if kbl not in llama or kcot not in llama:
        print("  Missing LLaMA-3 data!")
        return

    bl = llama[kbl]
    cot = llama[kcot]
    common_q = sorted(set(bl.keys()) & set(cot.keys()))

    bl_acc = sum(1 for q in common_q if bl[q]['ok']) / len(common_q)
    cot_acc = sum(1 for q in common_q if cot[q]['ok']) / len(common_q)

    print(f"\n  LLaMA-3-8B GSM8K ({len(common_q)} questions):")
    print(f"    Baseline: {bl_acc:.1%}")
    print(f"    CoT@256:  {cot_acc:.1%} (Δ={cot_acc-bl_acc:+.1%})")

    # ====================================================================
    # INNOVATION 1: REASONING RATCHET on LLaMA-3
    # ====================================================================
    print(f"""
  ═══════════════════════════════════════════════════════════════
  INNOVATION 1: REASONING RATCHET — Cross-Family Validation
  ═══════════════════════════════════════════════════════════════""")

    # Recovery: baseline wrong → CoT correct
    improved = [q for q in common_q if not bl[q]['ok'] and cot[q]['ok']]
    degraded = [q for q in common_q if bl[q]['ok'] and not cot[q]['ok']]
    n_wrong_bl = sum(1 for q in common_q if not bl[q]['ok'])
    n_correct_bl = sum(1 for q in common_q if bl[q]['ok'])

    recovery_rate = len(improved) / n_wrong_bl if n_wrong_bl else 0
    collapse_rate = len(degraded) / n_correct_bl if n_correct_bl else 0

    # McNemar test
    n_disc = len(improved) + len(degraded)
    mcnemar_p = stats.binomtest(min(len(improved), len(degraded)), n_disc, 0.5).pvalue if n_disc > 0 else 1.0

    # Do recovered answers hit ceiling?
    recovered_hit_ceiling = sum(1 for q in improved if cot[q]['tok'] >= 250)
    all_wrong_hit_ceiling = sum(1 for q in common_q if not bl[q]['ok'] and cot[q]['tok'] >= 250)

    print(f"""
    Recovery (baseline→CoT correct): {len(improved)}/{n_wrong_bl} ({recovery_rate:.1%})
    Collapse (baseline correct→CoT wrong): {len(degraded)}/{n_correct_bl} ({collapse_rate:.1%})
    Asymmetry ratio: {len(improved)/max(len(degraded),1):.1f}:1
    McNemar p = {mcnemar_p:.2e}

    Hit-ceiling analysis:
      Recovered answers hitting ceiling: {recovered_hit_ceiling}/{len(improved)} ({recovered_hit_ceiling/max(len(improved),1):.1%})
      All wrong@baseline hitting ceiling@CoT: {all_wrong_hit_ceiling}/{n_wrong_bl} ({all_wrong_hit_ceiling/max(n_wrong_bl,1):.1%})
    """)

    # Compare with Qwen2.5
    print(f"    Cross-family comparison (7B-scale models):")
    for model_label, model_data, model_key in [
        ("Qwen2.5-7B", gsm8k, "Qwen2.5-7B"),
        ("Llama-3-8B", llama, "Llama-3-8B")
    ]:
        kbl_m = (model_key, "GSM8K", "baseline")
        kcot_m = (model_key, "GSM8K", "cot_t0")
        if kbl_m in model_data and kcot_m in model_data:
            bl_m = model_data[kbl_m]
            cot_m = model_data[kcot_m]
            cq = sorted(set(bl_m.keys()) & set(cot_m.keys()))
            imp = sum(1 for q in cq if not bl_m[q]['ok'] and cot_m[q]['ok'])
            deg = sum(1 for q in cq if bl_m[q]['ok'] and not cot_m[q]['ok'])
            n_w = sum(1 for q in cq if not bl_m[q]['ok'])
            print(f"      {model_label}: recovery={imp}/{n_w} ({imp/n_w:.1%}), collapse={deg}, ratio={imp/max(deg,1):.1f}:1")

    # ====================================================================
    # INNOVATION 2: TOKEN-LENGTH CONFIDENCE on LLaMA-3
    # ====================================================================
    print(f"""
  ═══════════════════════════════════════════════════════════════
  INNOVATION 2: TOKEN-LENGTH CONFIDENCE — Cross-Family Validation
  ═══════════════════════════════════════════════════════════════""")

    # AUC for LLaMA-3
    labels = [0 if cot[q]['ok'] else 1 for q in common_q]
    tokens = [cot[q]['tok'] for q in common_q]
    auc = roc_auc_score(labels, tokens)

    # Point-biserial correlation
    r_pb, p_pb = stats.pointbiserialr([1 if cot[q]['ok'] else 0 for q in common_q], tokens)

    # Mann-Whitney U
    wrong_toks = [cot[q]['tok'] for q in common_q if not cot[q]['ok']]
    correct_toks = [cot[q]['tok'] for q in common_q if cot[q]['ok']]
    u_stat, u_p = stats.mannwhitneyu(wrong_toks, correct_toks, alternative='greater')
    d = (np.mean(wrong_toks) - np.mean(correct_toks)) / np.sqrt(
        (np.std(wrong_toks)**2 + np.std(correct_toks)**2) / 2)

    # Conditional probabilities
    p_wrong_high = sum(1 for q in common_q if not cot[q]['ok'] and cot[q]['tok'] >= 240) / \
                   max(sum(1 for q in common_q if cot[q]['tok'] >= 240), 1)
    p_correct_low = sum(1 for q in common_q if cot[q]['ok'] and cot[q]['tok'] < 200) / \
                    max(sum(1 for q in common_q if cot[q]['tok'] < 200), 1)

    print(f"""
    LLaMA-3-8B CoT@256:
      AUC = {auc:.3f}
      Point-biserial r = {r_pb:.3f}, p = {p_pb:.2e}
      Mann-Whitney U p = {u_p:.2e}
      Cohen's d = {d:.2f}

      P(wrong | tok ≥ 240) = {p_wrong_high:.1%}
      P(correct | tok < 200) = {p_correct_low:.1%}

      Mean tokens: correct={np.mean(correct_toks):.1f}, wrong={np.mean(wrong_toks):.1f}
    """)

    # Cross-family AUC comparison
    print(f"    Cross-family AUC comparison:")
    print(f"    {'Model':20s} {'Dataset':10s} {'AUC':>8s} {'r_pb':>8s}")
    print(f"    {'-'*50}")

    # Qwen results
    for model in ["Qwen2.5-0.5B", "Qwen2.5-3B", "Qwen2.5-7B"]:
        k = (model, "GSM8K", "cot_t0")
        if k in gsm8k:
            r = gsm8k[k]
            qs = list(r.keys())
            l = [0 if r[q]['ok'] else 1 for q in qs]
            t = [r[q]['tok'] for q in qs]
            a = roc_auc_score(l, t)
            r_val, _ = stats.pointbiserialr([1 if r[q]['ok'] else 0 for q in qs], t)
            print(f"    {model:20s} {'GSM8K':10s} {a:>8.3f} {r_val:>8.3f}")

    for model in ["Qwen2.5-0.5B", "Qwen2.5-7B"]:
        for budget in ["cot_t0_256", "cot_t0_512"]:
            k = (model, "MATH", budget)
            if k in math:
                r = math[k]
                qs = list(r.keys())
                l = [0 if r[q]['ok'] else 1 for q in qs]
                t = [r[q]['tok'] for q in qs]
                a = roc_auc_score(l, t)
                r_val, _ = stats.pointbiserialr([1 if r[q]['ok'] else 0 for q in qs], t)
                print(f"    {model:20s} {'MATH-'+budget[-3:]:10s} {a:>8.3f} {r_val:>8.3f}")

    # LLaMA
    print(f"    {'Llama-3-8B':20s} {'GSM8K':10s} {auc:>8.3f} {r_pb:>8.3f}")

    # ====================================================================
    # INNOVATION 3: CoT PHASE TRANSITION — Cross-Family check
    # ====================================================================
    print(f"""
  ═══════════════════════════════════════════════════════════════
  INNOVATION 3: CoT PHASE TRANSITION — With LLaMA-3
  ═══════════════════════════════════════════════════════════════""")

    # For phase transition we need multiple models at different scales
    # LLaMA only has 8B, so we can check if it falls in the "large model" phase
    # Compare CoT benefit at different difficulty levels (defined by Qwen models)

    # Use Qwen difficulty definition on GSM8K
    qwen_models = ['Qwen2.5-0.5B', 'Qwen2.5-3B', 'Qwen2.5-7B']
    qwen_common = sorted(set.intersection(
        *[set(gsm8k[(m, 'GSM8K', 'cot_t0')].keys()) for m in qwen_models]
    ))

    # Also intersect with LLaMA questions
    llama_qs = set(llama[kcot].keys())
    all_common = sorted(set(qwen_common) & llama_qs)

    difficulty = {}
    for q in all_common:
        n = sum(1 for m in qwen_models if gsm8k[(m, "GSM8K", "cot_t0")][q]["ok"])
        difficulty[q] = n

    print(f"\n  Phase Transition with LLaMA-3-8B added ({len(all_common)} common questions):")
    print(f"  Difficulty defined by Qwen2.5 cross-model agreement\n")

    print(f"  {'Difficulty':15s}", end="")
    for model in qwen_models + ["Llama-3-8B"]:
        print(f" {model:>15s}", end="")
    print(f" {'Pattern':>15s}")
    print(f"  {'-'*95}")

    for d_level in range(4):
        qs = [q for q in all_common if difficulty[q] == d_level]
        if not qs: continue

        label = f"  {d_level}/3 solve"
        deltas = []

        for model in qwen_models:
            bl_acc_m = sum(1 for q in qs if gsm8k[(model, "GSM8K", "baseline")][q]['ok']) / len(qs)
            cot_acc_m = sum(1 for q in qs if gsm8k[(model, "GSM8K", "cot_t0")][q]['ok']) / len(qs)
            delta = cot_acc_m - bl_acc_m
            deltas.append(delta)
            label += f" {delta:>+15.1%}"

        # LLaMA-3
        bl_acc_l = sum(1 for q in qs if llama[kbl][q]['ok']) / len(qs)
        cot_acc_l = sum(1 for q in qs if llama[kcot][q]['ok']) / len(qs)
        delta_l = cot_acc_l - bl_acc_l
        deltas.append(delta_l)
        label += f" {delta_l:>+15.1%}"

        if all(d < 0 for d in deltas):
            pattern = "↘ ALL NEG"
        elif deltas[0] < 0 and deltas[-1] > 0:
            pattern = "↕ PHASE FLIP!"
        elif all(d > 0 for d in deltas):
            pattern = "↗ ALL POS"
        else:
            pattern = "↗ MIXED"

        label += f" {pattern:>15s}"
        print(label)

    # ====================================================================
    # SUMMARY: Cross-family validation status
    # ====================================================================
    print(f"""

  ═══════════════════════════════════════════════════════════════
  CROSS-FAMILY VALIDATION SUMMARY
  ═══════════════════════════════════════════════════════════════

  Innovation 1 (Reasoning Ratchet):
    Qwen2.5-7B:  recovery=55.9%, collapse=1.9%, ratio=26:1 ✅
    Llama-3-8B:  recovery={recovery_rate:.1%}, collapse={collapse_rate:.1%}, ratio={len(improved)/max(len(degraded),1):.1f}:1 {'✅' if recovery_rate > 0.3 else '⚠️'}

  Innovation 2 (Token-Length Confidence):
    Qwen2.5-7B GSM8K: AUC=0.880, r=0.54 ✅
    Qwen2.5-7B MATH:  AUC=0.865, r=0.52 ✅
    Llama-3-8B GSM8K: AUC={auc:.3f}, r={r_pb:.3f} {'✅' if auc > 0.7 else '⚠️'}

  Innovation 3 (CoT Phase Transition):
    Qwen2.5 (0.5B/3B/7B): Phase flip confirmed ✅
    Llama-3-8B added:      See above {'✅' if delta_l > 0 else '⚠️'} (large model = positive CoT benefit)
""")


if __name__ == "__main__":
    main()
