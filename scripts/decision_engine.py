#!/usr/bin/env python3
"""
DECISION ENGINE: After 1024 experiment completes, determine Innovation 3 strategy.

If marginal efficiency (512→1024) > average efficiency → Accelerating Returns survives
If marginal efficiency (512→1024) < average efficiency → Switch to Phase Transition
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
    math = load_results("results_math_v2")

    print("=" * 80)
    print("DECISION ENGINE: Innovation 3 Strategy Based on 1024 Results")
    print("=" * 80)

    k256 = ("Qwen2.5-7B", "MATH", "cot_t0_256")
    k512 = ("Qwen2.5-7B", "MATH", "cot_t0_512")
    k1024 = ("Qwen2.5-7B", "MATH", "cot_t0_1024")

    if k1024 not in math:
        print("\n  1024 experiment not yet complete. Waiting...")
        # Show what we'll do when it's done
        print(f"""
  DECISION CRITERIA:
    1. Compute acc@1024, compute@1024
    2. Average efficiency = acc@1024 / compute@1024
    3. Marginal efficiency (512→1024) = (acc@1024 - acc@512) / (compute@1024 - compute@512)
    4. If marginal > average → Accelerating Returns INNOVATION VALID
    5. If marginal < average → Switch to Phase Transition

  Also check:
    - Recovery rate 512→1024 (wrong@512 → correct@1024)
    - Asymmetry ratio (recovery vs collapse at 512→1024)
    - AUC@1024 (token confidence signal at 1024)
""")
        return

    # Full analysis with 1024 data
    r256 = math[k256]
    r512 = math[k512]
    r1024 = math[k1024]
    common_q = sorted(set(r256.keys()) & set(r512.keys()) & set(r1024.keys()))

    acc_256 = sum(1 for q in common_q if r256[q]['ok']) / len(common_q)
    acc_512 = sum(1 for q in common_q if r512[q]['ok']) / len(common_q)
    acc_1024 = sum(1 for q in common_q if r1024[q]['ok']) / len(common_q)

    compute_256 = sum(r256[q]['tok'] for q in common_q)
    compute_512 = sum(r512[q]['tok'] for q in common_q)
    compute_1024 = sum(r1024[q]['tok'] for q in common_q)

    # Efficiencies
    avg_eff_256 = acc_256 / compute_256
    avg_eff_512 = acc_512 / compute_512
    avg_eff_1024 = acc_1024 / compute_1024

    marginal_256_512 = (acc_512 - acc_256) / (compute_512 - compute_256)
    marginal_512_1024 = (acc_1024 - acc_512) / (compute_1024 - compute_512)

    # Recovery analysis
    wrong_512 = [q for q in common_q if not r512[q]['ok']]
    recovered_512_1024 = sum(1 for q in wrong_512 if r1024[q]['ok'])
    recovery_rate_512_1024 = recovered_512_1024 / len(wrong_512) if wrong_512 else 0

    correct_512 = [q for q in common_q if r512[q]['ok']]
    collapsed_512_1024 = sum(1 for q in correct_512 if not r1024[q]['ok'])
    collapse_rate_512_1024 = collapsed_512_1024 / len(correct_512) if correct_512 else 0

    # McNemar
    improved = sum(1 for q in common_q if not r512[q]['ok'] and r1024[q]['ok'])
    degraded = sum(1 for q in common_q if r512[q]['ok'] and not r1024[q]['ok'])
    n_disc = improved + degraded
    mcnemar_p = stats.binomtest(min(improved, degraded), n_disc, 0.5).pvalue if n_disc > 0 else 1.0

    # AUC at 1024
    labels_1024 = [0 if r1024[q]['ok'] else 1 for q in common_q]
    tokens_1024 = [r1024[q]['tok'] for q in common_q]
    auc_1024 = roc_auc_score(labels_1024, tokens_1024)

    # Hit ceiling at 1024
    wrong_toks_1024 = [r1024[q]['tok'] for q in common_q if not r1024[q]['ok']]
    correct_toks_1024 = [r1024[q]['tok'] for q in common_q if r1024[q]['ok']]
    hit_1024_wrong = sum(1 for t in wrong_toks_1024 if t >= 1018) / len(wrong_toks_1024) if wrong_toks_1024 else 0

    print(f"""
  ═══════════════════════════════════════════════════════════════
  Qwen2.5-7B MATH: THREE-BUDGET COMPARISON ({len(common_q)} questions)
  ═══════════════════════════════════════════════════════════════

  Accuracy:
    256t:  {acc_256:.1%}
    512t:  {acc_512:.1%} (Δ={acc_512-acc_256:+.1%})
    1024t: {acc_1024:.1%} (Δ={acc_1024-acc_512:+.1%})

  Compute (total tokens):
    256t:  {compute_256:>8d}
    512t:  {compute_512:>8d} (+{compute_512-compute_256})
    1024t: {compute_1024:>8d} (+{compute_1024-compute_512})

  Average Efficiency (acc/compute):
    256t:  {avg_eff_256:.6f}
    512t:  {avg_eff_512:.6f} ({avg_eff_512/avg_eff_256:.2f}x vs 256)
    1024t: {avg_eff_1024:.6f} ({avg_eff_1024/avg_eff_256:.2f}x vs 256)

  Marginal Efficiency:
    256→512:  {marginal_256_512:.6f} ({marginal_256_512/avg_eff_256:.2f}x avg)
    512→1024: {marginal_512_1024:.6f} ({marginal_512_1024/avg_eff_256:.2f}x avg)

  Recovery (512→1024):
    Wrong@512→Correct@1024: {recovered_512_1024}/{len(wrong_512)} ({recovery_rate_512_1024:.1%})
    Correct@512→Wrong@1024: {collapsed_512_1024}/{len(correct_512)} ({collapse_rate_512_1024:.1%})
    Asymmetry: {recovered_512_1024/max(collapsed_512_1024,1):.1f}:1
    McNemar p = {mcnemar_p:.2e}

  Token Confidence:
    AUC@256:  0.865
    AUC@512:  0.719
    AUC@1024: {auc_1024:.3f}
    Hit ceiling@1024 (wrong): {hit_1024_wrong:.1%}
""")

    # DECISION
    print(f"  ═══════════════════════════════════════════════════════════════")
    print(f"  DECISION:")
    print(f"  ═══════════════════════════════════════════════════════════════")

    if marginal_512_1024 > avg_eff_256:
        print(f"    ✅ Marginal (512→1024) = {marginal_512_1024/avg_eff_256:.2f}x avg → ACCELERATING")
        print(f"    → Innovation 3 (Accelerating Returns) SURVIVES")
        print(f"    → Need to verify it's not just Ratchet corollary with stronger framing")
    elif marginal_512_1024 > avg_eff_1024:
        print(f"    ⚠️ Marginal (512→1024) = {marginal_512_1024/avg_eff_1024:.2f}x current avg → DECELERATING")
        print(f"    → Innovation 3 (Accelerating Returns) WEAKENS")
        print(f"    → Recommend switch to Phase Transition as Innovation 3")
    else:
        print(f"    ❌ Marginal (512→1024) = {marginal_512_1024:.6f} < avg = {avg_eff_1024:.6f}")
        print(f"    → DIMINISHING RETURNS at 1024")
        print(f"    → MUST switch to Phase Transition as Innovation 3")

    print(f"""
  RECOVERY PATTERN (three stages):
    256→512: {acc_512-acc_256:+.1%} (recovery rate: 55.9%)
    512→1024: {acc_1024-acc_512:+.1%} (recovery rate: {recovery_rate_512_1024:.1%})
    → Recovery rate is {'DECLINING' if recovery_rate_512_1024 < 0.559 else 'STABLE/INCREASING'}
    → This suggests {'approaching a natural ceiling' if recovery_rate_512_1024 < 0.3 else 'ongoing non-convergence'}
""")


if __name__ == "__main__":
    main()
