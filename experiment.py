"""
Committed-minority tipping experiment.

Stages:
  --stage calibrate : measure q_0, q_1 for each (regime) by running both
                      interpretations on the full corpus and judging outputs.
  --stage sweep     : run the agent-population dynamics across regimes x alphas
                      x seeds, saving per-round JSONL traces.

All GPT calls are cached to disk under cache/ (key = sha256 of model+prompt
JSON), so reruns are free.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI, APIError
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ---------- config --------------------------------------------------------

MARKER = "summarize"
T0 = "extract the three most important sentences verbatim from the article, preserving the article's exact wording"
T1 = "produce a brief paraphrased summary of the article in your own words, NOT copying phrases from the article"

INTERP = {"t0": T0, "t1": T1}

# Downstream tasks define which interpretation "wins".
# Worded so the FUNCTIONAL goal is clear, no format constraints that
# accidentally penalize the correct interpretation.
DOWNSTREAM = {
    "extractive": (
        "I am preparing a fact-check and need to copy the article's own "
        "wording directly into a document. I require the article's original "
        "phrasing verbatim — any paraphrase or rewording is useless to me "
        "because it changes the original language I need to quote."
    ),
    "gist": (
        "I have not read this article and I do not want to. I want it digested "
        "for me, in fresh everyday language, so I understand what happened "
        "without having to parse the article's own phrasing. A version that "
        "just lifts sentences from the article does not help me — I want it "
        "re-explained in someone else's words."
    ),
}

REGIMES = {
    # name : downstream_task_key
    "extractive": "extractive",
    "gist":       "gist",
}

MODEL_ACTOR    = "gpt-4o-mini"
MODEL_OBSERVER = "gpt-4o-mini"
MODEL_JUDGE    = "gpt-4o"

MAX_CONCURRENCY = 24
HARD_CALL_CAP = 200_000

# pricing (USD per 1M tokens) — approximations for cost-estimate printout
PRICE = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o":      {"in": 2.50, "out": 10.00},
}

# ---------- API client ----------------------------------------------------

def load_key() -> str:
    key_file = ROOT / "conf"
    key = key_file.read_text().strip().splitlines()[0].strip()
    if not key.startswith("sk-"):
        raise RuntimeError(f"conf does not look like an OpenAI key: {key[:8]}...")
    return key


CLIENT: AsyncOpenAI | None = None
SEM: asyncio.Semaphore | None = None
CALL_COUNT = {"n": 0, "in_tok": 0, "out_tok": 0, "cost": 0.0}

OLLAMA_CLIENT: AsyncOpenAI | None = None
OLLAMA_SEM: asyncio.Semaphore | None = None
OLLAMA_MODELS = {"qwen3:0.6b", "qwen3:1.7b", "qwen3:30b-a3b", "llama3.2:3b", "llama3.1:8b-instruct-q4_K_M"}


def init_client():
    global CLIENT, SEM, OLLAMA_CLIENT, OLLAMA_SEM
    CLIENT = AsyncOpenAI(api_key=load_key())
    SEM = asyncio.Semaphore(MAX_CONCURRENCY)
    OLLAMA_CLIENT = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    OLLAMA_SEM = asyncio.Semaphore(4)  # conservative for local inference


# ---------- cached chat call ---------------------------------------------

def _cache_path(key: str) -> Path:
    return CACHE / f"{key}.json"


def _cache_key(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps({"m": model, "s": system, "u": user, "t": temperature, "mx": max_tokens},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def chat(model: str, system: str, user: str, *, temperature: float = 0.0,
               max_tokens: int = 400, use_cache: bool = True) -> str:
    """Single-turn chat call. Cached when use_cache=True and temperature=0."""
    key = _cache_key(model, system, user, temperature, max_tokens)
    p = _cache_path(key)
    if use_cache and p.exists():
        return json.loads(p.read_text())["text"]

    if CALL_COUNT["n"] >= HARD_CALL_CAP:
        raise RuntimeError(f"hard call cap of {HARD_CALL_CAP} reached")

    async with SEM:
        for attempt in range(5):
            try:
                resp = await CLIENT.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user",   "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                in_tok = getattr(usage, "prompt_tokens", 0)
                out_tok = getattr(usage, "completion_tokens", 0)
                CALL_COUNT["n"] += 1
                CALL_COUNT["in_tok"]  += in_tok
                CALL_COUNT["out_tok"] += out_tok
                pr = PRICE.get(model, {"in": 0, "out": 0})
                CALL_COUNT["cost"] += in_tok/1e6*pr["in"] + out_tok/1e6*pr["out"]
                if use_cache:
                    p.write_text(json.dumps({"text": text, "in": in_tok, "out": out_tok}))
                return text
            except APIError as e:
                wait = 2 ** attempt
                print(f"  api error (attempt {attempt+1}): {e}; sleep {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
        raise RuntimeError("chat() exhausted retries")


# ---------- corpus --------------------------------------------------------

def load_corpus() -> list[dict]:
    with open(ROOT / "articles.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- actor / judge / observer prompts -----------------------------

ACTOR_SYSTEM_TMPL = (
    'You are an assistant whose current understanding of the instruction '
    '"{marker}" is the following: "{interp}". When asked to {marker} a text, '
    'you must follow that understanding exactly. Output ONLY the result of '
    'applying that understanding — no preamble, no commentary.'
)

ACTOR_USER_TMPL = 'Please {marker} the following article:\n\n{article}'

JUDGE_SYSTEM = (
    "You evaluate whether a candidate text would functionally serve a user's "
    "downstream goal. Judge fitness for purpose, not length or polish: would "
    "a reasonable person with this need find the candidate useful as-is? "
    "Reply with exactly one token: YES or NO. No other text."
)

JUDGE_USER_TMPL = (
    "USER'S DOWNSTREAM TASK:\n{task}\n\n"
    "CANDIDATE TEXT:\n{candidate}\n\n"
    "Is the candidate text USEFUL for the downstream task? Reply YES or NO."
)

OBSERVER_SYSTEM = (
    'You are a rational agent who is updating beliefs based on observed evidence. '
    'You currently interpret an instruction one way, and you just observed another '
    'agent use a different interpretation, with a specific outcome (success or '
    'failure on a shared downstream task). You must decide whether to switch to '
    'the observed interpretation. Reply with exactly one token: YES (switch) or '
    'NO (keep current). No other text.'
)

OBSERVER_USER_TMPL = (
    'The shared instruction is: "{marker}".\n'
    'Your current interpretation: "{own_interp}".\n'
    'You just watched another agent use this interpretation: "{other_interp}".\n'
    'Their output was judged {outcome} for the shared downstream task: '
    '"{downstream}".\n\n'
    'Based on this single observation, should you change YOUR interpretation '
    'of "{marker}" to match what they did? Reply YES or NO.'
)


async def actor_output(interp_name: str, article_text: str) -> str:
    sys_p = ACTOR_SYSTEM_TMPL.format(marker=MARKER, interp=INTERP[interp_name])
    usr_p = ACTOR_USER_TMPL.format(marker=MARKER, article=article_text)
    return (await chat(MODEL_ACTOR, sys_p, usr_p, temperature=0.0, max_tokens=300)).strip()


async def judge_score(downstream_task: str, candidate: str) -> int:
    usr_p = JUDGE_USER_TMPL.format(task=downstream_task, candidate=candidate)
    raw = (await chat(MODEL_JUDGE, JUDGE_SYSTEM, usr_p,
                      temperature=0.0, max_tokens=4)).strip().upper()
    return 1 if raw.startswith("Y") else 0


async def observer_decision(own_interp: str, other_interp: str, outcome: int,
                            downstream: str, temperature: float = 0.0,
                            obs_model: str = "gpt-4o-mini") -> bool:
    usr_p = OBSERVER_USER_TMPL.format(
        marker=MARKER,
        own_interp=INTERP[own_interp],
        other_interp=INTERP[other_interp],
        outcome="SUCCESSFUL" if outcome else "UNSUCCESSFUL",
        downstream=downstream,
    )
    if obs_model in OLLAMA_MODELS:
        return await _ollama_observer(obs_model, usr_p, temperature)
    # skip cache for temp>0 so each call is an independent sample
    use_cache = (temperature == 0.0)
    raw = (await chat(obs_model, OBSERVER_SYSTEM, usr_p,
                      temperature=temperature, max_tokens=4,
                      use_cache=use_cache)).strip().upper()
    return raw.startswith("Y")


async def _ollama_observer(model: str, usr_p: str, temperature: float) -> bool:
    is_qwen = model.startswith("qwen")
    # prepend /no_think for qwen3 thinking models
    prompt = f"/no_think\n{usr_p}" if is_qwen else usr_p
    extra = {"think": False} if is_qwen else {}
    async with OLLAMA_SEM:
        for attempt in range(4):
            try:
                resp = await OLLAMA_CLIENT.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": OBSERVER_SYSTEM},
                              {"role": "user",   "content": prompt}],
                    temperature=temperature,
                    max_tokens=500,
                    extra_body=extra or None,
                )
                raw = (resp.choices[0].message.content or "").strip().upper()
                return raw.startswith("Y")
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
    return False


# ---------- stage: calibrate ---------------------------------------------

async def run_calibration():
    corpus = load_corpus()
    print(f"[calib] {len(corpus)} articles, 2 interpretations, 2 regimes")
    print(f"[calib] ~{len(corpus)*2} actor calls + {len(corpus)*2*2} judge calls "
          f"= {len(corpus)*6} GPT calls (first run; cached thereafter)")

    # Step A: produce outputs for both interpretations on every article
    actor_tasks = []
    for art in corpus:
        for interp in ("t0", "t1"):
            actor_tasks.append((art["id"], interp, art["text"]))

    async def _one_actor(aid, interp, text):
        out = await actor_output(interp, text)
        return aid, interp, out

    actor_results = await tqdm_asyncio.gather(
        *[_one_actor(a, i, t) for (a, i, t) in actor_tasks],
        desc="actor outputs",
    )
    outputs = {(aid, interp): out for (aid, interp, out) in actor_results}

    # Step B: judge each output under each regime's downstream task
    judge_jobs = []
    for (aid, interp), out in outputs.items():
        for regime, ds_key in REGIMES.items():
            judge_jobs.append((aid, interp, regime, DOWNSTREAM[ds_key], out))

    async def _one_judge(aid, interp, regime, ds, out):
        score = await judge_score(ds, out)
        return aid, interp, regime, score

    judge_results = await tqdm_asyncio.gather(
        *[_one_judge(a, i, r, d, o) for (a, i, r, d, o) in judge_jobs],
        desc="judge scoring",
    )

    # Step C: aggregate
    by_regime: dict[str, dict[str, list[int]]] = {
        r: {"t0": [], "t1": []} for r in REGIMES
    }
    rows = []
    for (aid, interp, regime, score) in judge_results:
        by_regime[regime][interp].append(score)
        rows.append({"article_id": aid, "interpretation": interp,
                     "regime": regime, "success": score,
                     "output": outputs[(aid, interp)]})

    cal_dir = RESULTS / "calibration"
    cal_dir.mkdir(exist_ok=True)
    with (cal_dir / "raw.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # 20 sample triples for sanity check
    rng = random.Random(0)
    samples = rng.sample(rows, k=min(20, len(rows)))
    with (cal_dir / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    summary = {"models": {"actor": MODEL_ACTOR, "judge": MODEL_JUDGE,
                          "observer": MODEL_OBSERVER},
               "n_articles": len(corpus),
               "regimes": {}}
    for regime, dct in by_regime.items():
        q0 = float(np.mean(dct["t0"])) if dct["t0"] else float("nan")
        q1 = float(np.mean(dct["t1"])) if dct["t1"] else float("nan")
        # binomial 95% CI (Wilson) — simple normal approx for log
        def ci(succ_list):
            n = len(succ_list); p = np.mean(succ_list) if n else 0
            se = (p*(1-p)/max(n,1)) ** 0.5
            return [max(0.0, p - 1.96*se), min(1.0, p + 1.96*se)]
        summary["regimes"][regime] = {
            "q0_t0": q0,
            "q1_t1": q1,
            "q0_ci": ci(dct["t0"]),
            "q1_ci": ci(dct["t1"]),
            "predicted_alpha_star": (1 - q1/q0) if q0 > 0 and q1 < q0 else 0.0,
            "minority_is_useful": q1 >= q0,
        }
    summary["cost_so_far_usd"] = round(CALL_COUNT["cost"], 4)
    summary["calls"] = CALL_COUNT["n"]

    (cal_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== CALIBRATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {cal_dir}/samples.jsonl, raw.jsonl, summary.json")


# ---------- stage: sweep --------------------------------------------------

@dataclass
class RunConfig:
    regime: str
    alpha: float
    seed: int
    observer: str = "llm"   # "llm" | "mechanical" | "stochastic" | "ollama"
    N: int = 20
    T: int = 500
    obs_temp: float = 0.0
    obs_model: str = "gpt-4o-mini"  # model used when observer="ollama" or "llm"


ROLLING_WINDOW = 50  # for stochastic observer


async def run_one_sweep(cfg: RunConfig):
    corpus = load_corpus()
    rng = random.Random(cfg.seed * 1009 + int(cfg.alpha * 1000))

    n_committed = int(round(cfg.alpha * cfg.N))
    agents = []
    for i in range(cfg.N):
        if i < n_committed:
            agents.append({"interp": "t1", "committed": True})
        else:
            agents.append({"interp": "t0", "committed": False})

    ds_task = DOWNSTREAM[REGIMES[cfg.regime]]

    temp_tag = f"_temp{cfg.obs_temp}" if cfg.obs_temp != 0.0 else ""
    model_tag = f"_{cfg.obs_model.replace(':','_')}" if cfg.observer == "ollama" else ""
    out_dir = RESULTS / "sweep" / cfg.observer / f"N{cfg.N}_T{cfg.T}{temp_tag}{model_tag}" / cfg.regime
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"alpha={cfg.alpha:.2f}_seed={cfg.seed}.jsonl"

    # Rolling per-interp success history for the stochastic observer.
    succ_hist = {"t0": deque(maxlen=ROLLING_WINDOW),
                 "t1": deque(maxlen=ROLLING_WINDOW)}

    with trace_path.open("w") as f:
        f.write(json.dumps({"_config": asdict(cfg)}) + "\n")
        for t in range(cfg.T):
            i, j = rng.sample(range(cfg.N), 2)
            article = rng.choice(corpus)
            interp_i = agents[i]["interp"]
            out_i = await actor_output(interp_i, article["text"])
            success = await judge_score(ds_task, out_i)
            succ_hist[interp_i].append(success)

            switched = False
            adopt_p = None
            if not agents[j]["committed"]:
                interp_j = agents[j]["interp"]
                if interp_j != interp_i:
                    if cfg.observer in ("llm", "ollama"):
                        yes = await observer_decision(interp_j, interp_i,
                                                      success, ds_task,
                                                      temperature=cfg.obs_temp,
                                                      obs_model=cfg.obs_model)
                    elif cfg.observer == "mechanical":
                        # the rule the theorem's x*(alpha) formula assumes:
                        # switch iff the observed actor SUCCEEDED.
                        yes = bool(success)
                    elif cfg.observer == "stochastic":
                        # adopt with prob = rolling success rate of the OTHER
                        # interpretation. Falls back to current single
                        # observation if history is empty.
                        hist = succ_hist[interp_i]
                        p = (sum(hist) / len(hist)) if len(hist) > 0 else float(success)
                        adopt_p = p
                        yes = rng.random() < p
                    else:
                        raise ValueError(f"unknown observer: {cfg.observer}")
                    if yes:
                        agents[j]["interp"] = interp_i
                        switched = True

            maj_t1 = sum(1 for a in agents if not a["committed"] and a["interp"] == "t1")
            maj_size = sum(1 for a in agents if not a["committed"])
            x_t = maj_t1 / maj_size if maj_size else 0.0
            rec = {"t": t, "i": i, "j": j, "interp_i": interp_i,
                   "success": success, "switched": switched, "x_t": x_t}
            if adopt_p is not None:
                rec["adopt_p"] = adopt_p
            f.write(json.dumps(rec) + "\n")
    return cfg, trace_path


async def run_sweep(observer: str = "llm", regimes_filter: list[str] | None = None,
                    alphas: list[float] | None = None, N: int = 20, T: int = 500,
                    obs_temp: float = 0.0, obs_model: str = "gpt-4o-mini"):
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                  0.65, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.98, 0.99]
    seeds = [11, 22, 33]
    regs = regimes_filter or list(REGIMES.keys())
    cfgs = [RunConfig(regime=r, alpha=a, seed=s, observer=observer, N=N, T=T,
                      obs_temp=obs_temp, obs_model=obs_model)
            for r in regs for a in alphas for s in seeds]
    print(f"[sweep] {len(cfgs)} runs queued; calls ≈ {len(cfgs)*500*3} = "
          f"{len(cfgs)*500*3:,}")
    # Run a handful of full runs in parallel (each run is sequential internally)
    PARALLEL_RUNS = 6
    sem = asyncio.Semaphore(PARALLEL_RUNS)
    async def _bounded(c):
        async with sem:
            return await run_one_sweep(c)
    results = await tqdm_asyncio.gather(*[_bounded(c) for c in cfgs], desc="runs")
    print(f"[sweep] complete. calls={CALL_COUNT['n']}  cost≈${CALL_COUNT['cost']:.2f}")
    return results


# ---------- main ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["calibrate", "sweep"], required=True)
    p.add_argument("--observer", choices=["llm", "mechanical", "stochastic", "ollama"],
                   default="llm")
    p.add_argument("--regimes", nargs="*", default=None)
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--alphas", nargs="*", type=float, default=None)
    p.add_argument("--obs-temp", type=float, default=0.0)
    p.add_argument("--obs-model", type=str, default="gpt-4o-mini")
    args = p.parse_args()
    init_client()
    t0 = time.time()
    if args.stage == "calibrate":
        asyncio.run(run_calibration())
    else:
        asyncio.run(run_sweep(observer=args.observer, regimes_filter=args.regimes,
                              N=args.N, T=args.T, alphas=args.alphas,
                              obs_temp=args.obs_temp, obs_model=args.obs_model))
    print(f"\n[done] wall time {time.time()-t0:.1f}s  "
          f"calls={CALL_COUNT['n']}  cost≈${CALL_COUNT['cost']:.4f}")


if __name__ == "__main__":
    main()
