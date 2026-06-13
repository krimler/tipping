# committed-minority-tipping

Code for the paper "On the Tipping Behavior of LLM Agent Conventions".

## What this is

An agent-population simulation of committed-minority tipping dynamics in LLM
systems. Agents hold one of two interpretations of a shared capability marker
(`summarize`) and update based on observed task outcomes. A committed minority
is locked to an alternative interpretation. The experiment tests whether the
minority can tip the majority, and under what conditions.

Two regimes:
- **Extractive**: the downstream task requires verbatim extraction; the minority
  interpretation is harmful (q0 > q1, predicted threshold alpha* = 0.876).
- **Gist**: the downstream task requires paraphrased output; the minority
  interpretation is useful (q1 > q0, predicted threshold alpha* = 0).

Three observer rules: LLM (GPT-4o-mini YES/NO), mechanical (switch iff
observed success), stochastic (switch with probability = rolling success rate).

## Files

| File | Description |
|---|---|
| `experiment.py` | Calibration and population sweep |
| `analysis.py` | Loads results, generates plots and REPORT.md |
| `articles.jsonl` | 30 news articles used as the task corpus |
| `requirements.txt` | Python dependencies |
| `results/calibration/` | Measured q0, q1 per regime |
| `results/plots/` | All figures (9 PNG files) |
| `results/summary.json` | Empirical alpha* and RMSE per (regime, observer) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "sk-..." > conf          # OpenAI API key
```

Requires Python 3.10+. OpenAI API key must be in `./conf`.

## Running

```bash
# Calibrate q0, q1 (cached after first run, ~$0.07)
python experiment.py --stage calibrate

# Main sweep: all three observer rules
python experiment.py --stage sweep --observer llm       --N 50 --T 20000
python experiment.py --stage sweep --observer mechanical --N 50 --T 20000
python experiment.py --stage sweep --observer stochastic --N 50 --T 20000

# Generate plots and REPORT.md
python analysis.py
```

Temperature ablation (bypasses cache, ~$30):
```bash
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.3
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 0.7
python experiment.py --stage sweep --observer llm --N 50 --T 20000 --obs-temp 1.0
```

Local model observer (requires [Ollama](https://ollama.ai)):
```bash
ollama pull llama3.2:3b
python experiment.py --stage sweep --observer ollama --obs-model llama3.2:3b \
  --obs-temp 0.7 --N 50 --T 2000
```

## Caching

All OpenAI calls are cached to `cache/` keyed by SHA-256 of (model, prompt,
temperature, max_tokens). Reruns are free. The main sweeps are cache-saturated
after the first run because the 30-article corpus produces a small number of
unique prompts. Temperature ablation bypasses the cache (temperature is part
of the key).

## Cost

| Run | Cost |
|---|---|
| Calibration + main sweeps | ~$0.07 (cache-saturated) |
| Temperature ablation (3 temps x T=20000) | ~$30 |
| Local model (Ollama) | free |

The sweep traces (1.3 GB of JSONL) are excluded from this repo. Everything
else is included; plots and calibration data can be verified without rerunning.
