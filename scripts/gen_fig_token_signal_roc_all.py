#!/usr/bin/env python3
"""
Generate multi-model ROC curves for Token Count vs Log-probability.
Shows Qwen2.5-7B + Gemma-2-9B + LLaMA-3-8B for cross-family comparison.

Color scheme (Nature/Science academic):
  Qwen:  Blue family   (deep #1565C0 / medium #42A5F5)
  Gemma: Red family     (deep #C62828 / medium #EF5350)
  LLaMA: Teal family    (deep #00695C / medium #26A69A)

  Solid = Token count, Dashed = Log-probability
  MATH panel: 4 curves (Qwen+Gemma × token+logprob; no LLaMA MATH data)
  GSM8K panel: 6 curves (3 models × token+logprob)
"""
import json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
HUB = Path("/mnt/data2/zcz/tts_bench_optimized/results_v2")
LOGPROB_DIR = BASE / "results_v2" / "logprob_collection"
HUB_LOGPROB = Path("/mnt/data2/zcz/tts_bench_optimized/results_logprob")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Table 2 AUC values (paper-verified)
TABLE2 = {
    'Qwen2.5-7B_MATH_token': 0.865, 'Qwen2.5-7B_MATH_logprob': 0.561,
    'Qwen2.5-7B_GSM8K_token': 0.880, 'Qwen2.5-7B_GSM8K_logprob': 0.465,
    'Gemma-2-9B_MATH_token': 0.731, 'Gemma-2-9B_MATH_logprob': 0.584,
    'Gemma-2-9B_GSM8K_token': 0.699, 'Gemma-2-9B_GSM8K_logprob': 0.538,
    'LLaMA-3-8B_MATH_token': 0.687, 'LLaMA-3-8B_MATH_logprob': 0.682,
    'LLaMA-3-8B_GSM8K_token': 0.713, 'LLaMA-3-8B_GSM8K_logprob': 0.601,
}

MODELS = {
    'Qwen2.5-7B': {
        'tc': '#1565C0', 'lc': '#42A5F5',
        'tm': 'o', 'lm': 's',
        'lbl': 'Qwen2.5-7B',
    },
    'Gemma-2-9B': {
        'tc': '#C62828', 'lc': '#EF5350',
        'tm': '^', 'lm': 'D',
        'lbl': 'Gemma-2-9B',
    },
    'LLaMA-3-8B': {
        'tc': '#00695C', 'lc': '#26A69A',
        'tm': 'v', 'lm': 'p',
        'lbl': 'LLaMA-3-8B',
    },
}


def load_results(path, tok_field='tok'):
    with open(path) as f:
        d = json.load(f)
    results = d['results'] if isinstance(d, dict) and 'results' in d else d
    labels = np.array([1 if r['ok'] else 0 for r in results])
    tokens = np.array([r[tok_field] for r in results])
    return labels, tokens


def load_logprob(path):
    with open(path) as f:
        d = json.load(f)
    results = d if isinstance(d, list) else d['results']
    labels = np.array([1 if r['ok'] else 0 for r in results])
    logprobs = np.array([r['mean_logprob'] for r in results])
    return labels, logprobs


def roc_wrong(labels, scores):
    """ROC for detecting wrong answers (higher score = more likely wrong)."""
    wrong = 1 - labels
    fpr, tpr, _ = roc_curve(wrong, scores)
    return fpr, tpr, auc(fpr, tpr)


def thin(fpr, tpr, n=25):
    if len(fpr) <= n:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, n, dtype=int)
    return fpr[idx], tpr[idx]


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'legend.fontsize': 5.5, 'legend.framealpha': 0.92,
        'legend.edgecolor': '#CCCCCC',
    })

    curves = {}

    # === Qwen2.5-7B ===
    for ds in ['MATH', 'GSM8K']:
        # Token
        p = HUB / f"Qwen2.5-7B_{ds}_baseline_s42.json"
        if p.exists():
            labels, tokens = load_results(str(p), tok_field='tok')
            fpr, tpr, _ = roc_wrong(labels, tokens)
            curves[f'Qwen2.5-7B_{ds}_token'] = (fpr, tpr)

        # Logprob (from HUB results_logprob)
        ds_key = 'math' if ds == 'MATH' else 'gsm8k'
        p = HUB_LOGPROB / f"logprob_confidence_{ds_key}_7b.json"
        if p.exists():
            labels, lps = load_logprob(str(p))
            fpr, tpr, _ = roc_wrong(labels, -lps)
            curves[f'Qwen2.5-7B_{ds}_logprob'] = (fpr, tpr)

    # === Gemma-2-9B ===
    for ds in ['MATH', 'GSM8K']:
        p = HUB / f"gemma_controlled/Gemma-2-9B-IT_{ds}_base_256_s42.json"
        if p.exists():
            labels, tokens = load_results(str(p), tok_field='gen_tok')
            fpr, tpr, _ = roc_wrong(labels, tokens)
            curves[f'Gemma-2-9B_{ds}_token'] = (fpr, tpr)

        p = LOGPROB_DIR / f"gemma2_9b_{ds.lower()}_256.json"
        if p.exists():
            labels, lps = load_logprob(str(p))
            fpr, tpr, _ = roc_wrong(labels, -lps)
            curves[f'Gemma-2-9B_{ds}_logprob'] = (fpr, tpr)

    # === LLaMA-3-8B (MATH + GSM8K) ===
    for ds in ['MATH', 'GSM8K']:
        p = HUB / f"llama_controlled/{ds}_base_256.json"
        if p.exists():
            labels, tokens = load_results(str(p), tok_field='gen_tok')
            fpr, tpr, _ = roc_wrong(labels, tokens)
            curves[f'LLaMA-3-8B_{ds}_token'] = (fpr, tpr)

    # LLaMA GSM8K logprob (from HUB)
    p = HUB_LOGPROB / "logprob_confidence_gsm8k_llama.json"
    if p.exists():
        labels, lps = load_logprob(str(p))
        fpr, tpr, _ = roc_wrong(labels, -lps)
        curves['LLaMA-3-8B_GSM8K_logprob'] = (fpr, tpr)

    # LLaMA MATH logprob (from local logprob_collection)
    p = LOGPROB_DIR / "llama3_8b_math_256.json"
    if p.exists():
        labels, lps = load_logprob(str(p))
        fpr, tpr, _ = roc_wrong(labels, -lps)
        curves['LLaMA-3-8B_MATH_logprob'] = (fpr, tpr)

    print(f"Loaded {len(curves)} curves: {sorted(curves.keys())}")

    # === PLOT ===
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    for ds, ax in [('MATH', axes[0]), ('GSM8K', axes[1])]:
        ax.plot([0, 1], [0, 1], color='#888888', ls='--', lw=0.6, alpha=0.5, zorder=0)

        for mn in ['LLaMA-3-8B', 'Gemma-2-9B', 'Qwen2.5-7B']:
            sty = MODELS[mn]

            # Logprob (behind)
            kl = f'{mn}_{ds}_logprob'
            if kl in curves:
                fpr, tpr = curves[kl]
                pa = TABLE2.get(kl, auc(fpr, tpr))
                fm, tm = thin(fpr, tpr)
                ax.plot(fpr, tpr, color=sty['lc'], lw=1.5, ls='--', alpha=0.85, zorder=1)
                ax.plot(fm, tm, marker=sty['lm'], color=sty['lc'],
                        ms=4, lw=0, alpha=0.8, zorder=2,
                        label=f"{sty['lbl']} log-prob ({pa:.3f})")

            # Token (foreground)
            kt = f'{mn}_{ds}_token'
            if kt in curves:
                fpr, tpr = curves[kt]
                pa = TABLE2.get(kt, auc(fpr, tpr))
                fm, tm = thin(fpr, tpr)
                ax.plot(fpr, tpr, color=sty['tc'], lw=2.2,
                        solid_capstyle='round', alpha=0.95, zorder=3)
                ax.plot(fm, tm, marker=sty['tm'], color=sty['tc'],
                        ms=4, lw=0, alpha=0.8, zorder=4,
                        label=f"{sty['lbl']} token ({pa:.3f})")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.set_title(ds, fontweight='bold', fontsize=11)
        ax.legend(loc='lower right', fontsize=5.5, handlelength=1.5,
                  labelspacing=0.25, borderpad=0.4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linewidth=0.3)

    plt.tight_layout(w_pad=2.5)
    out = FIG_DIR / "fig_token_signal_roc"
    plt.savefig(f"{out}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out}.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out}.pdf / .png")
    plt.close()


if __name__ == "__main__":
    main()
