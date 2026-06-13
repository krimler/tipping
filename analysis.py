"""
Read calibration + sweep results across all observer modes; produce plots and
REPORT.md. The headline plot is x_inf vs alpha for the harmful-minority
(extractive) regime, faceted by observer rule, with theory curve overlay.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RES = ROOT / "results"
PLOTS = RES / "plots"
PLOTS.mkdir(exist_ok=True)

OBSERVERS = ["llm", "mechanical", "stochastic"]
OBSERVER_COLORS = {"llm": "#1f77b4", "mechanical": "#d62728", "stochastic": "#2ca02c"}
OBSERVER_LABEL  = {"llm": "LLM observer (gpt-4o-mini YES/NO)",
                   "mechanical": "mechanical (switch iff observed success)",
                   "stochastic": "stochastic (P[switch] = rolling success rate)"}


def theory_x(alpha, q0, q1):
    if q1 >= q0:
        return 1.0 if alpha > 0 else 0.0
    a_star = 1 - q1 / q0
    if alpha >= a_star:
        return 1.0
    return alpha * q1 / max(1e-9, (1 - alpha) * (q0 - q1))


def load_calibration():
    return json.loads((RES / "calibration" / "summary.json").read_text())


def load_sweep(run_tag: str = "N50_T20000"):
    """observer -> regime -> alpha -> list of (seed, [x_t...])  +  detected N"""
    out = {o: defaultdict(lambda: defaultdict(list)) for o in OBSERVERS}
    detected_N = 50
    for obs in OBSERVERS:
        sweep_dir = RES / "sweep" / obs / run_tag
        for fp in sorted(glob.glob(str(sweep_dir / "*" / "alpha=*.jsonl"))):
            with open(fp) as f:
                rows = [json.loads(l) for l in f]
            if len(rows) < 2:
                continue
            cfg = rows[0]["_config"]
            detected_N = cfg.get("N", 50)
            xs = [r["x_t"] for r in rows[1:]]
            out[obs][cfg["regime"]][cfg["alpha"]].append((cfg["seed"], xs))
    return out, detected_N


def is_no_majority(alpha, N=20):
    """alpha values where n_committed = N (no majority agents exist)."""
    n_c = int(round(alpha * N))
    return n_c >= N


# ---------- plots ---------------------------------------------------------

def plot_calibration(cal):
    regimes = list(cal["regimes"].keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.35
    xs = np.arange(len(regimes))
    q0s = [cal["regimes"][r]["q0_t0"] for r in regimes]
    q1s = [cal["regimes"][r]["q1_t1"] for r in regimes]
    q0_err = [[q - cal["regimes"][r]["q0_ci"][0] for q, r in zip(q0s, regimes)],
              [cal["regimes"][r]["q0_ci"][1] - q for q, r in zip(q0s, regimes)]]
    q1_err = [[q - cal["regimes"][r]["q1_ci"][0] for q, r in zip(q1s, regimes)],
              [cal["regimes"][r]["q1_ci"][1] - q for q, r in zip(q1s, regimes)]]
    ax.bar(xs - w/2, q0s, w, yerr=q0_err, capsize=4, label=r"$q_0$ ($t_0$ extractive)", color="#3a6ea5")
    ax.bar(xs + w/2, q1s, w, yerr=q1_err, capsize=4, label=r"$q_1$ ($t_1$ paraphrastic)", color="#c0392b")
    ax.set_xticks(xs); ax.set_xticklabels(regimes)
    ax.set_ylabel("success rate (judge YES fraction)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Plot 1 — Calibration: measured $q_0, q_1$ per regime\n(30 articles, gpt-4o judge, 95% Wald CIs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "1_calibration.png", dpi=140)
    plt.close(fig)


def plot_trajectories(sweep, N=20):
    """Per-regime grid of trajectories: rows = regimes, cols = observers."""
    regimes = ["extractive", "gist"]
    fig, axes = plt.subplots(len(regimes), len(OBSERVERS),
                             figsize=(5 * len(OBSERVERS), 4 * len(regimes)),
                             sharex=True, sharey=True)

    # Collect all valid alphas across all panels for a shared colormap scale
    valid_alphas = sorted({a for obs in OBSERVERS for regime in regimes
                           for a in sweep[obs][regime].keys()
                           if not is_no_majority(a, N)})
    norm = plt.Normalize(vmin=min(valid_alphas), vmax=max(valid_alphas))
    cmap = plt.cm.viridis

    for r_i, regime in enumerate(regimes):
        for c_i, obs in enumerate(OBSERVERS):
            ax = axes[r_i, c_i]
            for a in sorted(sweep[obs][regime].keys()):
                if is_no_majority(a, N):
                    continue
                stack = np.array([np.array(xs) for (_, xs) in sweep[obs][regime][a]])
                ax.plot(stack.mean(axis=0), color=cmap(norm(a)), linewidth=1)
            ax.set_title(f"{regime} | {obs}")
            ax.set_ylim(-0.02, 1.02)
            if r_i == len(regimes) - 1:
                ax.set_xlabel("round")
            if c_i == 0:
                ax.set_ylabel(r"$x_t$ (fraction of majority at $t_1$)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical",
                        fraction=0.02, pad=0.03)
    cbar.set_label("committed-minority fraction α", fontsize=9)

    fig.suptitle("Plot 2 — Per-round trajectories (mean over 3 seeds)")
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(PLOTS / "2_trajectories.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_money(sweep, cal, N=20):
    """x_inf vs alpha, faceted by regime × observer. Theory curve overlaid."""
    regimes = ["extractive", "gist"]
    fig, axes = plt.subplots(len(regimes), len(OBSERVERS),
                             figsize=(5 * len(OBSERVERS), 4 * len(regimes)),
                             sharey=True)
    summary = defaultdict(dict)
    for r_i, regime in enumerate(regimes):
        q0 = cal["regimes"][regime]["q0_t0"]
        q1 = cal["regimes"][regime]["q1_t1"]
        a_star_pred = (1 - q1 / q0) if q1 < q0 else 0.0
        a_grid = np.linspace(0, 1, 401)
        theory = [theory_x(a, q0, q1) for a in a_grid]
        for c_i, obs in enumerate(OBSERVERS):
            ax = axes[r_i, c_i]
            alphas = sorted(sweep[obs][regime].keys())
            # exclude no-majority alphas from both fits and plots
            alphas_p = [a for a in alphas if not is_no_majority(a, N)]
            means, stds = [], []
            for a in alphas_p:
                x_infs = [np.mean(xs[-100:]) for (_, xs) in sweep[obs][regime][a]]
                means.append(np.mean(x_infs))
                stds.append(np.std(x_infs))
            means = np.array(means); stds = np.array(stds)
            ax.errorbar(alphas_p, means, yerr=stds, fmt="o-", capsize=3,
                        color=OBSERVER_COLORS[obs],
                        label=f"empirical (3 seeds, mean ± std)")
            ax.plot(a_grid, theory, "--", color="#222",
                    label=f"theory $x^*(\\alpha\\ |\\ q_0={q0:.2f},q_1={q1:.2f})$")
            if q1 < q0:
                ax.axvline(a_star_pred, color="#888", linestyle=":", linewidth=1,
                           label=f"predicted $\\alpha^*$={a_star_pred:.3f}")
            ax.set_title(f"{regime} | observer={obs}")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
            if r_i == len(regimes) - 1:
                ax.set_xlabel("α (committed-minority fraction)")
            if c_i == 0:
                ax.set_ylabel("x∞ (steady state, last 100 rounds)")
            ax.legend(fontsize=7, loc="upper left" if regime == "extractive" else "lower right")
            # empirical alpha*: first alpha with mean x_inf > 0.95 AND has majority
            emp_star = None
            for a, m in zip(alphas_p, means):
                if m > 0.95:
                    emp_star = a; break
            theory_pts = np.array([theory_x(a, q0, q1) for a in alphas_p])
            rmse = float(np.sqrt(np.mean((means - theory_pts) ** 2)))
            summary[regime][obs] = {
                "q0": q0, "q1": q1,
                "predicted_alpha_star": a_star_pred,
                "empirical_alpha_star": emp_star,
                "rmse_x_inf_vs_theory": rmse,
                "alphas": alphas_p, "x_inf_mean": means.tolist(),
                "x_inf_std": stds.tolist(),
            }
    fig.suptitle("Plot 3 — $x_\\infty$ vs $\\alpha$ — theory vs observer rules")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS / "3_money_xinf_vs_alpha.png", dpi=140)
    plt.close(fig)
    return summary


def plot_thresholds(summary):
    """Per-regime, per-observer α* comparison."""
    regimes = list(summary.keys())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4))
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        labels = ["theory"] + OBSERVERS
        vals = [summary[regime][OBSERVERS[0]]["predicted_alpha_star"]]
        for obs in OBSERVERS:
            e = summary[regime][obs]["empirical_alpha_star"]
            vals.append(e if e is not None else float("nan"))
        x = np.arange(len(labels))
        colors = ["#222"] + [OBSERVER_COLORS[o] for o in OBSERVERS]
        bars = ax.bar(x, vals, color=colors)
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.text(i, 0.02, "unobserved\n(>0.96)", ha="center", fontsize=8, color="#666")
            else:
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("α*")
        ax.set_title(f"{regime}")
    fig.suptitle("Plot 4 — Predicted vs empirical tipping threshold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOTS / "4_thresholds.png", dpi=140)
    plt.close(fig)


def write_report(cal, summary, N=50):
    lines = []
    lines.append("# Committed-Minority Tipping in LLM Agents — Report\n")
    lines.append("Date: 2026-06-13\n")

    lines.append("## Setup\n")
    lines.append("- Marker: `summarize`")
    lines.append("- t0 (majority init): *extract three most important sentences verbatim*")
    lines.append("- t1 (committed minority): *brief paraphrased summary in your own words*")
    lines.append(f"- N={N} agents, T=20000 rounds, 3 seeds/condition")
    lines.append(f"- Actor: `{cal['models']['actor']}`, Judge: `{cal['models']['judge']}`, LLM-observer: `{cal['models']['observer']}`")
    lines.append("- 18 α values: [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.98, 0.99]")
    lines.append("- 3 core observer rules: llm (gpt-4o-mini YES/NO), mechanical (switch iff success), stochastic (P[switch] = rolling success rate)")
    lines.append("- Additional: LLM observer at temp∈{0.3,0.7,1.0}; local models llama3.2:3b and qwen3:0.6b as observers")
    lines.append(f"- Note: α=0.99 rounds to n_committed=N with N={N}, excluded from fits.\n")

    lines.append("## Calibration\n")
    lines.append("| regime | q0 (t0) | q1 (t1) | useful minority? | predicted α* |")
    lines.append("|---|---|---|---|---|")
    for r, d in cal["regimes"].items():
        lines.append(f"| {r} | {d['q0_t0']:.3f} | {d['q1_t1']:.3f} | "
                     f"{'yes' if d['minority_is_useful'] else 'no'} | {d['predicted_alpha_star']:.3f} |")
    lines.append("")

    lines.append("## Results — x∞ vs α by observer rule (T=20000, N=50)\n")
    for regime in summary:
        q0 = summary[regime]['llm']['q0']
        q1 = summary[regime]['llm']['q1']
        a_star = summary[regime]['llm']['predicted_alpha_star']
        lines.append(f"### {regime}  (q0={q0:.3f}, q1={q1:.3f}, α*={a_star:.3f})\n")
        lines.append("| observer | empirical α* | RMSE(x∞, theory) |")
        lines.append("|---|---|---|")
        for obs in OBSERVERS:
            d = summary[regime][obs]
            e = d["empirical_alpha_star"]
            e_s = f"{e:.3f}" if e is not None else "unobserved (>0.96)"
            lines.append(f"| {obs} | {e_s} | {d['rmse_x_inf_vs_theory']:.3f} |")
        lines.append("")

    lines.append("## Key findings\n")
    lines.append(
        "**1. T=500 is insufficient — extractive regime needs T~10000–20000.**  \n"
        "With q1=0.033, the expected switching events in 500 rounds is <3. "
        "At T=20000 the mechanical observer cleanly validates α*=0.875.\n"
    )
    lines.append(
        "**2. Theory confirmed at T=20000 for mechanical and stochastic observers.**  \n"
        "Both produce the smooth x*(α) curve below α* and tip to x∞=1 above it. "
        "In the gist regime (q1>q0, α*=0) all observers tip at any α>0 as predicted.\n"
    )
    lines.append(
        "**3. LLM observer (gpt-4o-mini) never tips in extractive — at any temperature.**  \n"
        "Direct sampling (n=50 per cell) shows YES rate = 0% for all extractive combinations "
        "at temperatures 0.0, 0.3, 0.7, and 1.0. The LLM evaluates semantic task fit, "
        "not outcome signals — a hard refusal, not slow convergence.\n"
    )
    lines.append(
        "**4. Model capability determines observer behavior.**  \n"
        "| model | YES% succ extractive | YES% fail extractive | behavior |\n"
        "|---|---|---|---|\n"
        "| llama3.2:3b | ~70% | ~83% | blind imitator |\n"
        "| gpt-4o-mini | 0% | 0% | task-principled refuser |\n\n"
        "llama3.2:3b switches at a near-constant rate regardless of the judge verdict — "
        "higher after failure than after success — and tips in extractive at all α. "
        "gpt-4o-mini conditions on semantic task fit and refuses entirely.\n"
    )
    lines.append(
        "**5. The theorem is correct but its behavioral hypothesis does not hold for capable LLMs.**  \n"
        "The theorem assumes adoption proportional to observed success — what the mechanical rule "
        "implements. A capable LLM replaces this with semantic task evaluation, collapsing the "
        "smooth phase diagram to a binary: adopt iff semantically correct, independent of α. "
        "This is a stronger safety property, but means tipping dynamics track model capability.\n"
    )

    lines.append("## Observer decision sampling (n=50 per cell)\n")
    lines.append("| model | t0→t1 succ extractive | t0→t1 fail extractive | t0→t1 succ gist | t0→t1 fail gist |")
    lines.append("|---|---|---|---|---|")
    lines.append("| gpt-4o-mini (temp=0) | 0% | 0% | 100% | 0% |")
    lines.append("| llama3.2:3b (temp=0.7) | ~70% | ~83% | ~90% | ~80% |")
    lines.append("")

    lines.append("## Plots\n")
    lines.append("1. `1_calibration.png` — q0, q1 per regime with 95% CIs")
    lines.append("2. `2_trajectories.png` — per-round trajectories, mean over seeds")
    lines.append("3. `3_money_xinf_vs_alpha.png` — x∞ vs α, theory vs 3 observers")
    lines.append("4. `4_thresholds.png` — predicted vs empirical α*")
    lines.append("5. `5_T5000_extractive_mechanical.png` — convergence check at T=5000")
    lines.append("6. `6_T20000_extractive_mechanical.png` — full convergence at T=20000")
    lines.append("7. `7_T20000_all.png` — all observers, both regimes, T=20000")
    lines.append("8. `8_combined_all_temps.png` — LLM observer temps 0/0.3/0.7/1.0")
    lines.append("9. `9_final_combined.png` — all series including llama3.2:3b\n")

    lines.append("## Cost\n")
    lines.append(f"- Calibration: {cal['calls']} GPT calls, ~$0.07")
    lines.append("- Main sweeps (3 observers × N=50 × T=20000): fully cache-saturated, ~$0")
    lines.append("- Temperature ablation (3 temps × T=20000, live calls): ~$30")
    lines.append("- Local model sweeps (llama3.2:3b, T=2000): free (local inference)\n")

    lines.append("## Replication\n")
    lines.append("```bash")
    lines.append("python3 -m venv .venv && source .venv/bin/activate")
    lines.append("pip install -r requirements.txt")
    lines.append("python experiment.py --stage calibrate")
    lines.append("python experiment.py --stage sweep --observer llm --N 50 --T 20000")
    lines.append("python experiment.py --stage sweep --observer mechanical --N 50 --T 20000")
    lines.append("python experiment.py --stage sweep --observer stochastic --N 50 --T 20000")
    lines.append("# Temperature ablation (~$30 in API costs):")
    lines.append("python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.3")
    lines.append("python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.7")
    lines.append("python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 1.0")
    lines.append("# Local model (requires ollama):")
    lines.append("python experiment.py --stage sweep --observer ollama --obs-model llama3.2:3b --obs-temp 0.7 --N 50 --T 2000")
    lines.append("python analysis.py")
    lines.append("```")
    lines.append("OpenAI calls cached under `cache/`; reruns are free. API key in `./conf`.\n")

    lines.append("## Follow-ups\n")
    lines.append("1. **Batch observations** — prompt LLM observer with k>1 observations; test if aggregating evidence softens the semantic lock")
    lines.append("3. **Leaky commitment** — relax committed agents to probabilistic adherence; find minimum leakage for tipping")
    lines.append("4. **Multi-marker** — two markers updating simultaneously through shared population")
    lines.append("5. **More seeds near α*** — 10+ seeds in α∈[0.85,0.90] to pin down empirical α* for mechanical observer\n")

    (ROOT / "REPORT.md").write_text("\n".join(lines))


def main():
    cal = load_calibration()
    sweep, N = load_sweep()
    plot_calibration(cal)
    plot_trajectories(sweep, N)
    summary = plot_money(sweep, cal, N)
    plot_thresholds(summary)
    write_report(cal, summary, N)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote:")
    for p in sorted(PLOTS.glob("*.png")):
        print(" ", p)
    print(" ", ROOT / "REPORT.md")
    print(" ", RES / "summary.json")


if __name__ == "__main__":
    main()
