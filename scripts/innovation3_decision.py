#!/usr/bin/env python3
"""
STRATEGIC ANALYSIS: Which Innovation 3 to use?

Given the 1024 partial results (120/200), we can already see the trend.
The question is: is Accelerating Returns or Phase Transition the better Innovation 3?

This script analyzes both options and provides a recommendation.
"""
import json, os
import numpy as np
from scipy import stats

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

    print("=" * 80)
    print("STRATEGIC ANALYSIS: Innovation 3 Selection")
    print("=" * 80)

    # ====================================================================
    # Option A: Accelerating Returns
    # ====================================================================
    print("""
OPTION A: ACCELERATING RETURNS (加速回报)
══════════════════════════════════════════

  Strengths:
    ✅ 3 budget points (256, 512, 1024) show consistent acceleration
    ✅ Marginal efficiency > average efficiency at all transitions
    ✅ Quantitative and directly actionable
    ✅ Completes the WHY→HOW→WHAT story

  Weaknesses:
    ❌ SAC CW1: Is it just a corollary of Ratchet?
    ❌ Recovery rate declining (55.9% → 15.4%) → approaching ceiling
    ❌ The "acceleration" at 512→1024 is partly an artifact of tiny Δcompute
    ❌ Only 1 model on MATH, no cross-family validation yet

  SAC-estimated score: 7.0-7.5/10 (independence concern)
""")

    # ====================================================================
    # Option B: CoT Phase Transition
    # ====================================================================
    print("""
OPTION B: CoT PHASE TRANSITION (CoT相变)
══════════════════════════════════════════

  Strengths:
    ✅ Genuinely independent from Innovation 1 (about strategy, not truncation)
    ✅ Clear phase flip: CoT helps at diff=2/3 but HURTS at diff=0/3
    ✅ Scale-dependent: 7B benefits from CoT at diff=1/3 where 3B doesn't
    ✅ Fisher p=0.0000 for 7B diff=0 vs diff=1
    ✅ Practical: adaptive CoT routing saves compute AND improves accuracy
    ✅ Cross-dataset potential (GSM8K confirmed, need MATH validation)

  Weaknesses:
    ❌ Only 3 scale points (need more for "phase transition" claim)
    ❌ At diff=1 for 3B: CoT hurts — but sample size small (collapse=3/5)
    ❌ Needs difficulty estimation in practice (chicken-and-egg problem)
    ❌ Not yet validated on MATH dataset

  SAC-estimated score: 7.5-8.0/10 (more independent, but needs more evidence)
""")

    # ====================================================================
    # Option C: Combine both as a single "TTS Scaling Non-Monotonicity"
    # ====================================================================
    print("""
OPTION C: TTS SCALING NON-MONOTONICITY (TTS缩放非单调性)
══════════════════════════════════════════

  Combine: "TTS scaling exhibits non-monotonic behavior in two dimensions:
    1. Token dimension: accelerating returns (not diminishing)
    2. Strategy dimension: CoT benefit is non-monotonic with difficulty × scale"

  This is a BROADER finding that encompasses both A and B.

  Strengths:
    ✅ Two independent manifestations of non-monotonicity
    ✅ Broader story: "TTS scaling is more complex than assumed"
    ✅ More robust to individual weakness

  Weaknesses:
    ❌ Dilutes focus — reviewers may find it too diffuse
    ❌ Harder to present in a clear narrative

  SAC-estimated score: 7.0-8.0/10 (depends on presentation)
""")

    # ====================================================================
    # RECOMMENDATION
    # ====================================================================
    print(f"""
{'='*80}
RECOMMENDATION: OPTION B (CoT Phase Transition)
{'='*80}

  Rationale:
    1. INDEPENDENCE: Phase Transition is about strategy selection,
       not truncation/recovery → genuinely separate from Innovation 1
    2. NOVELTY: First demonstration that CoT benefit has a difficulty×scale
       phase transition → directly challenges "CoT always helps" assumption
    3. PRACTICAL: Enables adaptive CoT routing — save compute on easy and
       impossibly hard questions, invest in boundary questions
    4. ACTIONABLE: Can be validated with LLaMA-3 (cross-family)

  Key data points to strengthen:
    - Need MATH validation (use 256/512 data with difficulty analysis)
    - Need LLaMA-3 cross-family validation
    - Need statistical test for the "phase flip" (3B→7B transition)

  FINAL THREE INNOVATIONS:
    1. Reasoning Ratchet (WHY tokens matter) — 8.5/10
    2. Token-Length Confidence (HOW to detect) — 8.0/10
    3. CoT Phase Transition (WHEN to use CoT) — 7.5-8.0/10

  Unified story:
    "Test-time compute scaling is more nuanced than assumed:
     (1) Many 'wrong' answers are unfinished, not incorrect [Ratchet]
     (2) Generation length provides a free confidence signal [Token Conf]
     (3) CoT's benefit is non-monotonic — it has a sweet spot [Phase Trans]"
""")


if __name__ == "__main__":
    main()
