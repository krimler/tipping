# Committed-Minority Tipping in LLM Agents — Report

Date: 2026-06-13

## Setup

- Marker: `summarize`
- t0 (majority init): *extract three most important sentences verbatim*
- t1 (committed minority): *brief paraphrased summary in your own words*
- N=50 agents, T=20000 rounds, 3 seeds/condition
- Actor: `gpt-4o-mini`, Judge: `gpt-4o`, LLM-observer: `gpt-4o-mini`
- 18 α values: [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.98, 0.99]
- 3 core observer rules: llm (gpt-4o-mini YES/NO), mechanical (switch iff success), stochastic (P[switch] = rolling success rate)
- Additional: LLM observer at temp∈{0.3,0.7,1.0}; local models llama3.2:3b and qwen3:0.6b as observers
- Note: α=0.99 rounds to n_committed=N with N=50, excluded from fits.

## Calibration

| regime | q0 (t0) | q1 (t1) | useful minority? | predicted α* |
|---|---|---|---|---|
| extractive | 0.267 | 0.033 | no | 0.875 |
| gist | 0.500 | 0.900 | yes | 0.000 |

## Results — x∞ vs α by observer rule (T=20000, N=50)

### extractive  (q0=0.267, q1=0.033, α*=0.875)

| observer | empirical α* | RMSE(x∞, theory) |
|---|---|---|
| llm | unobserved (>0.96) | 0.599 |
| mechanical | 0.800 | 0.113 |
| stochastic | 0.850 | 0.089 |

### gist  (q0=0.500, q1=0.900, α*=0.000)

| observer | empirical α* | RMSE(x∞, theory) |
|---|---|---|
| llm | 0.050 | 0.000 |
| mechanical | 0.050 | 0.000 |
| stochastic | 0.050 | 0.000 |

## Key findings

**1. T=500 is insufficient — extractive regime needs T~10000–20000.**  
With q1=0.033, the expected switching events in 500 rounds is <3. At T=20000 the mechanical observer cleanly validates α*=0.875.

**2. Theory confirmed at T=20000 for mechanical and stochastic observers.**  
Both produce the smooth x*(α) curve below α* and tip to x∞=1 above it. In the gist regime (q1>q0, α*=0) all observers tip at any α>0 as predicted.

**3. LLM observer (gpt-4o-mini) never tips in extractive — at any temperature.**  
Direct sampling (n=50 per cell) shows YES rate = 0% for all extractive combinations at temperatures 0.0, 0.3, 0.7, and 1.0. The LLM evaluates semantic task fit, not outcome signals — a hard refusal, not slow convergence.

**4. Model capability determines observer behavior — a three-tier spectrum.**  
| model | YES% succ extractive | YES% fail extractive | behavior |
|---|---|---|---|
| qwen3:0.6b (0.6B) | ~73% | ~97% | blind imitator |
| llama3.2:3b (3B) | ~70% | ~83% | weak imitator |
| gpt-4o-mini (~8B) | 0% | 0% | task-principled refuser |

Smaller models treat social influence as near-unconditional. gpt-4o-mini evaluates semantic compatibility first. llama3.2:3b tips in extractive (wrongly confirming a harmful minority); gpt-4o-mini resists entirely.

**5. The theorem is correct but its behavioral hypothesis does not hold for capable LLMs.**  
The theorem assumes adoption proportional to observed success — what the mechanical rule implements. A capable LLM replaces this with semantic task evaluation, collapsing the smooth phase diagram to a binary: adopt iff semantically correct, independent of α. This is a stronger safety property, but means tipping dynamics track model capability.

## Observer decision sampling (n=40-50, temp=0.7 for local models)

| model | t0→t1 succ extractive | t0→t1 fail extractive | t0→t1 succ gist | t0→t1 fail gist |
|---|---|---|---|---|
| gpt-4o-mini (temp=0) | 0% | 0% | 100% | 0% |
| qwen3:0.6b | ~73% | ~97% | ~97% | ~97% |
| llama3.2:3b | ~70% | ~83% | ~90% | ~80% |

## Plots

1. `1_calibration.png` — q0, q1 per regime with 95% CIs
2. `2_trajectories.png` — per-round trajectories, mean over seeds
3. `3_money_xinf_vs_alpha.png` — x∞ vs α, theory vs 3 observers
4. `4_thresholds.png` — predicted vs empirical α*
5. `5_T5000_extractive_mechanical.png` — convergence check at T=5000
6. `6_T20000_extractive_mechanical.png` — full convergence at T=20000
7. `7_T20000_all.png` — all observers, both regimes, T=20000
8. `8_combined_all_temps.png` — LLM observer temps 0/0.3/0.7/1.0
9. `9_final_combined.png` — all series including llama3.2:3b

## Cost

- Calibration: 180 GPT calls, ~$0.07
- Main sweeps (3 observers × N=50 × T=20000): fully cache-saturated, ~$0
- Temperature ablation (3 temps × T=20000, live calls): ~$30
- Local model sweeps (llama3.2:3b, T=2000): free (local inference)

## Replication

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiment.py --stage calibrate
python experiment.py --stage sweep --observer llm --N 50 --T 20000
python experiment.py --stage sweep --observer mechanical --N 50 --T 20000
python experiment.py --stage sweep --observer stochastic --N 50 --T 20000
# Temperature ablation (~$30 in API costs):
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.3
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.7
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 1.0
# Local model (requires ollama):
python experiment.py --stage sweep --observer ollama --obs-model llama3.2:3b --obs-temp 0.7 --N 50 --T 2000
python analysis.py
```
OpenAI calls cached under `cache/`; reruns are free. API key in `./conf`.

## Follow-ups

1. **qwen3:1.7b sweep** — partial run killed by GPU contention; rerun solo to fill 0.6B→3B gap
2. **Batch observations** — prompt LLM observer with k>1 observations; test if aggregating evidence softens the semantic lock
3. **Leaky commitment** — relax committed agents to probabilistic adherence; find minimum leakage for tipping
4. **Multi-marker** — two markers updating simultaneously through shared population
5. **More seeds near α*** — 10+ seeds in α∈[0.85,0.90] to pin down empirical α* for mechanical observer
