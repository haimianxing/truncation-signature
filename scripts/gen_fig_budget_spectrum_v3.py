#!/usr/bin/env python3
"""
Generate Budget Spectrum Figure (fig_budget_spectrum_v3).
Dual-model accuracy curves + ceiling rate shading + three-zone heuristic.

Extracted from conversation history (2026-04-26).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def main():
    # Standard 7B data (Qwen2.5-7B MATH)
    std_budgets = [128, 256, 384, 512, 768, 1024]
    std_acc = [5.0, 53.5, 76.0, 78.5, 80.5, 80.5]
    std_ceil = [99.5, 55.5, 11.0, 2.5, 0.0, 0.0]

    # R1-Distill-7B data
    r1_budgets = [256, 512, 1024, 2048, 4096, 8192]
    r1_acc = [12.5, 63.5, 80.0, 80.5, 80.5, 80.5]
    r1_ceil = [96.0, 35.5, 3.0, 0.5, 0.5, 0.5]

    color_std = '#3498DB'
    color_r1 = '#E74C3C'

    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    ax1.plot(std_budgets, std_acc, 'o-', color=color_std, linewidth=2,
             markersize=5, label='Qwen2.5-7B Accuracy')
    ax1.plot(r1_budgets, r1_acc, 's-', color=color_r1, linewidth=2,
             markersize=5, label='R1-Distill-7B Accuracy')

    ax1.set_xlabel('Token Budget')
    ax1.set_ylabel('Accuracy (%)', color='black')
    ax1.set_xscale('log', base=2)
    ax1.set_xlim(90, 10000)
    ax1.set_ylim(0, 90)
    ax1.axhline(y=80.5, color='gray', linestyle=':', alpha=0.5, label='Peak (80.5%)')

    # Ceiling rate on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(std_budgets, [-c for c in std_ceil], width=30, alpha=0.15,
            color=color_std, label='Qwen Ceiling')
    ax2.bar(r1_budgets, [-c for c in r1_ceil], width=80, alpha=0.15,
            color=color_r1, label='R1 Ceiling')
    ax2.set_ylabel('Ceiling Rate (%)', color='gray')
    ax2.set_ylim(-110, 10)
    ax2.tick_params(axis='y', labelcolor='gray')

    # Three-zone annotations
    ax1.axvspan(90, 350, alpha=0.05, color='red', label='TRUNCATION')
    ax1.axvspan(350, 600, alpha=0.05, color='orange', label='INFLECTION')
    ax1.axvspan(600, 10000, alpha=0.05, color='green', label='EFFICIENT')

    ax1.text(180, 3, 'TRUNCATION', fontsize=7, color='red', alpha=0.7, ha='center')
    ax1.text(470, 3, 'INFLECTION', fontsize=7, color='orange', alpha=0.7, ha='center')
    ax1.text(2000, 3, 'EFFICIENT', fontsize=7, color='green', alpha=0.7, ha='center')

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1[:5], labels1[:5], loc='center right', fontsize=7)

    ax1.set_title('Token Budget Spectrum: Standard vs. Reasoning Model')
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        outpath = FIG_DIR / f"fig_budget_spectrum_v3.{ext}"
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close()


if __name__ == "__main__":
    main()
