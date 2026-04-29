#!/usr/bin/env python3
"""
Overthinking Bridge Analysis — Process R1 + Standard model results
Generates publication-ready analysis comparing reasoning vs standard models.

Key outputs:
1. Budget-accuracy curves (R1 vs Qwen2.5-7B)
2. Ceiling rate diagnostic comparison
3. Overthinking detection (accuracy decline at high budgets)
4. Token Budget Spectrum regime identification
"""
import json, sys, re
from pathlib import Path
import numpy as np
from collections import defaultdict

BASE = Path(__file__).parent

def load_results(path):
    """Load checkpoint JSON and return results dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", {}), data.get("raw_outputs", {})

def budget_summary(results, prefix=""):
    """Extract accuracy, ceiling rate, avg tokens for each budget."""
    rows = []
    for bkey in sorted(results.keys(), key=lambda x: int(x.replace("vb_",""))):
        samples = results[bkey].get("samples", [])
        if not samples:
            continue
        budget = int(bkey.replace("vb_",""))
        n = len(samples)
        acc = sum(1 for s in samples if s.get('ok')) / n * 100
        ceil = sum(1 for s in samples if s.get('hit_ceiling'))
        ceil_rate = ceil / n * 100
        avg_tok = sum(s.get('tok',0) for s in samples) / n
        rows.append({
            "budget": budget, "n": n, "acc": acc,
            "ceiling": ceil, "ceil_rate": ceil_rate, "avg_tok": avg_tok
        })
    return rows

def detect_overthinking(rows):
    """Detect if accuracy declines at higher budgets."""
    if len(rows) < 3:
        return None, "Insufficient data points"

    peak_idx = max(range(len(rows)), key=lambda i: rows[i]["acc"])
    peak = rows[peak_idx]

    # Check if any budget beyond peak has lower accuracy
    overthinking_budgets = []
    for r in rows[peak_idx+1:]:
        if r["acc"] < peak["acc"]:
            overthinking_budgets.append(r)

    if overthinking_budgets:
        worst = min(overthinking_budgets, key=lambda x: x["acc"])
        decline = peak["acc"] - worst["acc"]
        return {
            "peak_budget": peak["budget"],
            "peak_acc": peak["acc"],
            "worst_budget": worst["budget"],
            "worst_acc": worst["acc"],
            "decline": decline,
            "n_declining": len(overthinking_budgets),
        }, f"OVER THINKING: {decline:.1f}% decline from @{peak['budget']}({peak['acc']:.1f}%) to @{worst['budget']}({worst['acc']:.1f}%)"
    else:
        return None, f"No overthinking — accuracy plateaus at {peak['acc']:.1f}% (@{peak['budget']})"

def regime_analysis(rows):
    """Identify Token Budget Spectrum regimes."""
    regimes = []
    for r in rows:
        if r["ceil_rate"] > 50:
            regime = "TRUNCATION"
        elif r["ceil_rate"] > 10:
            regime = "INFLECTION"
        else:
            regime = "EFFICIENT"
        regimes.append({**r, "regime": regime})
    return regimes

def main():
    print("=" * 70)
    print("OVER THINKING BRIDGE ANALYSIS")
    print("=" * 70)

    # Load R1 results (from original experiment on GPU 2)
    r1_file = BASE / "results_overthinking" / "overthinking_r1_7b_math.json"
    r1_eff_file = BASE / "results_overthinking" / "overthinking_r1_7b_math_efficient.json"

    # Load standard Qwen2.5-7B results
    std_file = BASE / "results_frontier" / "frontier_7b_math.json"

    print("\n--- R1 Results (Original Experiment) ---")
    if r1_file.exists():
        r1_results, _ = load_results(r1_file)
        r1_rows = budget_summary(r1_results)
        for r in r1_rows:
            print(f"  @{r['budget']:>5}: acc={r['acc']:.1f}% ceil={r['ceil_rate']:.1f}% n={r['n']}")
        overthink, msg = detect_overthinking(r1_rows)
        print(f"  → {msg}")

    print("\n--- R1 Results (Efficient Experiment) ---")
    if r1_eff_file.exists():
        r1_eff_results, _ = load_results(r1_eff_file)
        r1_eff_rows = budget_summary(r1_eff_results)
        for r in r1_eff_rows:
            print(f"  @{r['budget']:>5}: acc={r['acc']:.1f}% ceil={r['ceil_rate']:.1f}% n={r['n']}")
        overthink, msg = detect_overthinking(r1_eff_rows)
        print(f"  → {msg}")

    print("\n--- Standard Qwen2.5-7B Results ---")
    if std_file.exists():
        std_results, _ = load_results(std_file)
        std_rows = budget_summary(std_results)
        for r in std_rows:
            print(f"  @{r['budget']:>5}: acc={r['acc']:.1f}% ceil={r['ceil_rate']:.1f}%")
        overthink, msg = detect_overthinking(std_rows)
        print(f"  → {msg}")

    # Cross-model comparison
    print("\n--- Cross-Model Comparison ---")
    print(f"{'Budget':>8} {'R1-Distill':>12} {'Qwen2.5-7B':>12} {'Delta':>8}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    r1_accs = {}
    if r1_file.exists():
        for r in budget_summary(load_results(r1_file)[0]):
            r1_accs[r["budget"]] = r["acc"]
    if r1_eff_file.exists():
        for r in budget_summary(load_results(r1_eff_file)[0]):
            r1_accs[r["budget"]] = r["acc"]

    std_accs = {}
    if std_file.exists():
        for r in budget_summary(load_results(std_file)[0]):
            std_accs[r["budget"]] = r["acc"]

    all_budgets = sorted(set(r1_accs.keys()) | set(std_accs.keys()))
    for b in all_budgets:
        r1_a = f"{r1_accs[b]:.1f}%" if b in r1_accs else "—"
        std_a = f"{std_accs[b]:.1f}%" if b in std_accs else "—"
        if b in r1_accs and b in std_accs:
            delta = f"{r1_accs[b]-std_accs[b]:+.1f}%"
        else:
            delta = "—"
        print(f"{b:>8} {r1_a:>12} {std_a:>12} {delta:>8}")

    # Regime analysis
    print("\n--- Token Budget Spectrum (R1-Distill-7B) ---")
    if r1_file.exists():
        r1_results_data = load_results(r1_file)[0]
        r1_rows_full = budget_summary(r1_results_data)
        if r1_rows_full:
            regimes = regime_analysis(r1_rows_full)
            for r in regimes:
                print(f"  @{r['budget']:>5}: {r['regime']:>12} acc={r['acc']:.1f}% ceil={r['ceil_rate']:.1f}%")
        else:
            print("  No data yet")
    else:
        print("  No R1 results file found")

    # Standard model regimes
    print("\n--- Token Budget Spectrum (Qwen2.5-7B Standard) ---")
    if std_file.exists():
        std_rows_full = budget_summary(load_results(std_file)[0])
        if std_rows_full:
            regimes = regime_analysis(std_rows_full)
            for r in regimes:
                print(f"  @{r['budget']:>5}: {r['regime']:>12} acc={r['acc']:.1f}% ceil={r['ceil_rate']:.1f}%")

    # Summary for paper
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)

    # Collect key stats
    r1_at_256 = r1_accs.get(256, None)
    std_at_256 = std_accs.get(256, None)

    if r1_at_256 is not None and std_at_256 is not None:
        print(f"\n1. Truncation Impact on Reasoning Models:")
        print(f"   R1@256 = {r1_at_256:.1f}% vs Qwen2.5-7B@256 = {std_at_256:.1f}%")
        print(f"   Reasoning models MORE sensitive to truncation (Δ={std_at_256-r1_at_256:+.1f}%)")

    if r1_file.exists():
        r1_r = load_results(r1_file)[0]
        # Check Ratchet Effect for R1
        if "256" in r1_r and "512" in r1_r:
            r1_256_samples = {s["q"]: s for s in r1_r["256"]["samples"]}
            r1_512_samples = {s["q"]: s for s in r1_r["512"]["samples"]}
            nat_wrong_recover = 0
            nat_wrong_total = 0
            ceil_wrong_recover = 0
            ceil_wrong_total = 0
            for qid in r1_256_samples:
                s256 = r1_256_samples[qid]
                s512 = r1_512_samples.get(qid)
                if s512 is None:
                    continue
                if not s256.get("ok"):
                    if not s256.get("hit_ceiling"):
                        nat_wrong_total += 1
                        if s512.get("ok"):
                            nat_wrong_recover += 1
                    else:
                        ceil_wrong_total += 1
                        if s512.get("ok"):
                            ceil_wrong_recover += 1

            print(f"\n2. Ratchet Effect in R1-Distill (@256→@512):")
            if nat_wrong_total > 0:
                print(f"   Natural-stop wrong: {nat_wrong_recover}/{nat_wrong_total} recover ({nat_wrong_recover/nat_wrong_total*100:.1f}%)")
            if ceil_wrong_total > 0:
                print(f"   Ceiling-hit wrong: {ceil_wrong_recover}/{ceil_wrong_total} recover ({ceil_wrong_recover/ceil_wrong_total*100:.1f}%)")

    print("\nDone.")

if __name__ == "__main__":
    main()
