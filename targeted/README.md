# Targeted benchmarks

Each benchmark here stress-tests a single system resource. This page covers what they share and each
benchmark's own README covers the specific workload, any additional setup requirement, and its metrics.

- [`paper_summarization`](paper_summarization/) — concurrency
- [`bug_localization`](bug_localization/) — storage I/O
- [`semantic_embedding`](semantic_embedding/) — CPU compute

## Setup

Every benchmark is a directory with two scripts and its own dependencies.

```bash
cd targeted/<benchmark>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs Agency at a fixed commit; pulls PyTorch

python prepare.py                 # everything before the profiling: datasets, images, checks
python benchmark.py               # the measured run
```

`prepare.py` is idempotent, i.e. each step skips if already done. Run it again after `--force` or the agency framework changed.

## Model access

Export your provider's `base_url`, `api_key` and `model` before running any benchmark:

```bash
export LLM_BASE_URL=https://<host>/v1
export LLM_API_KEY=<key>
export LLM_MODEL=<model>              # the id your provider uses
export LLM_REASONING_EFFORT=high      # optional; only for a thinking model
```

Anything serving the OpenAI chat-completions API works — a hosted provider, a gateway, or a local
server. To use a provider's own API instead, pass `--backend anthropic`, or `--backend bedrock` with
`--region`; both take the key from the same place.

The key may also be a single-line file, which is what to use inside a container where an environment
variable is visible in `docker inspect`. First one that exists wins:

1. `$AGENCY_BENCH_API_KEY_FILE`
2. `/run/secrets/api-key`
3. `api-key` at the suite root (`../../api-key` from here)

## Common flags

All three `benchmark.py` accept these. Everything else is benchmark-specific.

| Flag | Default | Purpose |
|---|---|---|
| `--base-url` | `$LLM_BASE_URL` | The endpoint |
| `--model` | `$LLM_MODEL`, else `gpt-5.6-luna` | The model driving the agent |
| `--reasoning-effort` | `$LLM_REASONING_EFFORT` | `low`, `medium`, `high`. Unset sends nothing, i.e. the provider's default |
| `--backend` | `openai` | `openai`, `bedrock`, `anthropic` or `vllm` |
| `--harness` | `native` | Agency's own ReAct loop. External harnesses will be supported soon. |
| `--region` | `us-east-2` | AWS Bedrock region; `--backend bedrock` only |
| `--context-limit` | `196000` | Tokens |
| `--max-steps` | varies | Per-agent turn cap. |
| `--out` | `output/` | Where traces and results land |

`BENCH_BACKEND`, `BENCH_HARNESS` and `BENCH_REGION` override the corresponding flags, for scripted
sweeps.

## Output

```
output/<harness>/<model>/
  <case>/
    profiler/*.pt.trace.json   # Chrome trace; you can open in ui.perfetto.dev
    profiler/summary.{json,md} # profiler metrics
    result.json                # this benchmark's metrics of interest, plus the full run config
    agent_logs/                # per-agent conversation and tool timeline
  results.{json,md}            # aggregate over every case in the run
```

Paths are keyed by harness and model, with `/` and `:` replaced by `_`. `<case>` is whatever the benchmark varies, e.g., a topic and width, a corpus variant, a SWE-bench instance.

## Reading the results

- **The trace is the result.** A benchmark also scores the agent's answer, but that is a sanity signal — the metrics of interest for each benchmark are found in the trace and `summary.md`.
- **Focus on the `sandbox_metrics`.** The `sandbox_metrics` is scoped to the agent's sandbox, and the `workload_metrics` focuses on the driver process on the host side.
- **Keep the host quiet.** Competing load on the host adds noise during the profiling.

A guide to adding a benchmark is coming soon. Until then, copy [`paper_summarization`](paper_summarization/)
— it is the closest to a template — and open an issue.
