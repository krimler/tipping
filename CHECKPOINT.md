# Experiment Checkpoint Log

Replication record for the committed-minority tipping experiment.

## Environment

- Host: macOS Darwin 24.6.0 (arm64)
- Python: 3.14.4 (`/opt/homebrew/bin/python3`)
- Working dir: project root (clone of repo)
- Venv: `./.venv` (activate with `source .venv/bin/activate`)
- OpenAI API key: read from `./conf`
- Ollama: local inference server at `http://localhost:11434` (qwen3:0.6b, qwen3:1.7b, llama3.2:3b)

## Models

- Actor: `gpt-4o-mini` (OpenAI, cached)
- Judge: `gpt-4o` (OpenAI, cached)
- LLM observer (default): `gpt-4o-mini` (OpenAI)
- Local observers: `qwen3:0.6b`, `qwen3:1.7b`, `llama3.2:3b` (Ollama, not cached)

## Pinned deps

openai 2.36.0, numpy 2.4.4, matplotlib 3.10.9, tqdm 4.67.3

## Progress

- [x] Calibration — gist: q0=0.50, q1=0.90; extractive: q0=0.27, q1=0.03, α*=0.875
- [x] Sweep N=50, T=20000 — all 3 observers × 2 regimes × 18 α × 3 seeds (cache-saturated)
- [x] T=500 shown insufficient for extractive; T=20000 confirms theory for mechanical/stochastic
- [x] LLM temperature ablation (temp=0/0.3/0.7/1.0) — YES rate = 0% for extractive at all temps
- [x] Local model sweep — llama3.2:3b (T=2000); qwen3:0.6b YES rate sampled directly
- [x] Plots 1–9 + REPORT.md updated

## Key results (2026-06-13)

| observer | extractive x∞ | gist x∞ | behavior |
|---|---|---|---|
| mechanical | follows theory, tips above α*=0.875 | tips at α>0 | theory-matching |
| stochastic | smooth partial-mixture | tips at α>0 | theory-matching |
| LLM gpt-4o-mini (any temp) | stuck at 0 | tips at α≥0.05 | semantic refusal |
| llama3.2:3b | tips at all α (wrongly) | tips at α>0 | blind imitation |
| qwen3:0.6b | tips at all α (wrongly) | tips at α>0 | blind imitation |

**Model capability determines observer behavior**: small models are blind imitators;
capable LLMs evaluate semantic task fit and refuse incorrect adoption regardless of α or temperature.

## Total cost

- Calibration + main sweeps: ~$0.07 (cache-saturated after first run)
- Temperature ablation (3 temps × T=20000, live calls): ~$30
- Local model inference: free
