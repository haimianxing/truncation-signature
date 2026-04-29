#!/usr/bin/env python3
"""
Generate Ratchet Effect Figure (Updated for 13 conditions across 3 families × 3 tasks)
Shows 0/521 natural-stop vs 54.5% ceiling-hit recovery across all conditions.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Data from comprehensive analysis (all 15 conditions at 256→512 transition)
CONDITIONS = [
    # (Family, Model, Dataset, NatW, CeilW, CeilR)
    ("Qwen2.5", "0.5B", "MATH", 22, 127, 37),
    ("Qwen2.5", "3B", "MATH", 14, 115, 72),
    ("Qwen2.5", "7B", "MATH", 20, 101, 66),
    ("Qwen2.5", "3B", "GSM8K", 9, 108, 84),
    ("Qwen2.5", "7B", "GSM8K", 12, 93, 74),
    ("Gemma-2", "2B", "MATH", 70, 21, 0),
    ("Gemma-2", "2B", "GSM8K", 69, 18, 3),
    ("Gemma-2", "9B", "MATH", 47, 18, 4),
    ("Gemma-2", "9B", "GSM8K", 26, 13, 7),
    ("LLaMA-3", "8B", "GSM8K", 43, 0, 0),
    ("R1-Distill", "7B", "MATH", 38, 2, 2),
    ("R1-Distill", "7B", "GSM8K", 0, 152, 88),
    ("Qwen2.5", "7B", "HumanEval", 99, 35, 1),
]

# Colors for families
FAMILY_COLORS = {
    "Qwen2.5": "#2563EB",   # Blue
    "Gemma-2": "#DC2626",   # Red
    "LLaMA-3": "#059669",   # Green
    "R1-Distill": "#9333EA", # Purple
}

FAMILY_HATCH = {
    "Qwen2.5": "",
    "Gemma-2": "//",
    "LLaMA-3": "..",
    "R1-Distill": "xx",
}

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]})

    # === Left panel: Per-condition recovery rates ===
    labels = []
    ceil_recovery_rates = []
    nat_recovery_rates = []
    colors = []
    hatches = []

    for family, model, dataset, natw, ceilw, ceilr in CONDITIONS:
        label = f"{model}\n{dataset}"
        labels.append(label)
        ceil_pct = (ceilr / ceilw * 100) if ceilw > 0 else 0
        ceil_recovery_rates.append(ceil_pct)
        nat_recovery_rates.append(0.0)  # Always 0
        colors.append(FAMILY_COLORS[family])
        hatches.append(FAMILY_HATCH[family])

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    # Ceiling-hit recovery bars
    bars_ceil = ax1.barh(y_pos + bar_height/2, ceil_recovery_rates, bar_height,
                          color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                          label='Ceiling-hit recovery rate')

    # Add value labels on ceiling bars
    for i, (val, family, model, dataset, natw, ceilw, ceilr) in enumerate(
        zip(ceil_recovery_rates, *[list(x) for x in zip(*CONDITIONS)])):
        if val > 0:
            ax1.text(val + 1, y_pos[i] + bar_height/2, f'{val:.0f}%',
                    va='center', fontsize=8, fontweight='bold')
        else:
            ax1.text(1, y_pos[i] + bar_height/2, '0%',
                    va='center', fontsize=8, color='gray')

    # Natural-stop recovery bars (all zero)
    bars_nat = ax1.barh(y_pos - bar_height/2, nat_recovery_rates, bar_height,
                         color='#9CA3AF', alpha=0.4, edgecolor='black', linewidth=0.5,
                         label='Natural-stop recovery rate')

    # Add "0%" and count labels for natural-stop
    for i, (family, model, dataset, natw, ceilw, ceilr) in enumerate(CONDITIONS):
        ax1.text(1, y_pos[i] - bar_height/2, f'0/{natw}',
                va='center', fontsize=7, color='#6B7280', style='italic')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Recovery Rate (%)', fontsize=11)
    ax1.set_xlim(0, 100)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Add family separators
    prev_family = CONDITIONS[0][0]
    for i, (family, *_) in enumerate(CONDITIONS):
        if family != prev_family and i > 0:
            ax1.axhline(y=y_pos[i] - 0.5, color='gray', linewidth=0.5, linestyle='--')
        prev_family = family

    # Family labels on the left
    family_ranges = {}
    for i, (family, *_) in enumerate(CONDITIONS):
        if family not in family_ranges:
            family_ranges[family] = [i, i]
        else:
            family_ranges[family][1] = i

    for family, (start, end) in family_ranges.items():
        mid = (y_pos[start] + y_pos[end]) / 2
        ax1.text(-8, mid, family, ha='center', va='center', fontsize=9,
                fontweight='bold', color=FAMILY_COLORS[family], rotation=90)

    ax1.set_title('Per-Condition Recovery Rates', fontsize=12, fontweight='bold')

    # === Right panel: Pooled summary ===
    total_natw = sum(c[3] for c in CONDITIONS)
    total_natr = 0  # Always 0
    total_ceilw = sum(c[4] for c in CONDITIONS)
    total_ceilr = sum(c[5] for c in CONDITIONS)
    total_ceil_pct = total_ceilr / total_ceilw * 100 if total_ceilw > 0 else 0

    categories = ['Natural-stop\n(Type B)', 'Ceiling-hit\n(Type A)']
    rates = [0, total_ceil_pct]
    bar_colors = ['#9CA3AF', '#2563EB']
    counts = [f'0/{total_natw}', f'{total_ceilr}/{total_ceilw}']

    bars = ax2.bar(categories, rates, color=bar_colors, alpha=0.85,
                    edgecolor='black', linewidth=0.8, width=0.6)

    for bar, rate, count in zip(bars, rates, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., max(height, 3) + 2,
                f'{rate:.1f}%\n({count})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Recovery Rate (%)', fontsize=11)
    ax2.set_ylim(0, 85)
    ax2.set_title('Pooled (13 conditions)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # Add Fisher p-value annotation
    ax2.text(0.5, 0.02, r"Fisher's exact $p < 10^{-99}$",
            transform=ax2.transAxes, ha='center', fontsize=10,
            style='italic', color='#DC2626',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', alpha=0.8))

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        outpath = FIG_DIR / f"fig_ratchet_effect.{ext}"
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
