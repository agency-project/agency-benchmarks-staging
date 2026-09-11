# Agency Benchmarks

Systems-performance benchmarks for agentic workflows built on
[Agency](https://github.com/agency-project/agency-staging).

## Benchmarks

### Targeted

Each targted benchmark focuses on stress-testing a single system resource.

| Benchmark | Dimension | Dataset | Extra setup | Cheapest run |
|---|---|---|---|---|
| [`paper_summarization`](targeted/paper_summarization/) | Concurrency | TREC-COVID | `agency-sandbox` image | `--topics 1 --widths 1` |
| [`bug_localization`](targeted/bug_localization/) | Storage I/O | SWE-bench Verified | ~25 GB, passwordless `sudo` | `--limit 1` |
| [`semantic_embedding`](targeted/semantic_embedding/) | CPU | CNN/DailyMail | ~7 GB | `--variant short` |
| `short_qa` | Memory and cache | HotpotQA | — | To be released |

### End-to-end

Each benchmark is an end-to-end agentic application built from several agents and tools, so it stresses many
dimensions at once. Being refined before public release.

| Benchmark | Dimensions |
|---|---|
| Recursive self-improvement harness | CPU, memory, storage I/O, concurrency, network |
| Job application | CPU, memory, concurrency |
| Hardware generation | CPU, end-to-end loops |

## Requirements

- Linux on x86_64 with cgroup v2. The profiler reads `/proc` and `/sys/fs/cgroup`; macOS and ARM are
  not supported.
- Python 3.11+.
- Docker.
- An OpenAI-compatible model endpoint, as a `base_url`, an `api_key` and a `model`. Every run makes
  billed LLM calls, one per agent step.

## Quickstart

Build Agency's sandbox image from the Agency Staging Repo:

```bash
git clone https://github.com/agency-project/agency-staging.git
git -C agency-staging checkout d66f5363cf32d5b7c00bd669a5a584a1c5eb5786
(cd agency-staging && ./images/build.sh)
```

Then run the cheapest benchmark, for example, paper summarization with one topic at width 1 and two agent runs:

```bash
export LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export LLM_API_KEY=<key>
export LLM_MODEL=gpt-5

cd targeted/paper_summarization
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare.py
python benchmark.py --topics 1 --widths 1
```

## Results

```
output/native/gpt-5/1/W1/t1/
  profiler/*.pt.trace.json     open in ui.perfetto.dev
  profiler/summary.{json,md}   profiler metrics, machine- and human-readable
  result.json                  metrics and the full run config
  agent_logs/                  per-agent conversation and tool timeline
```

The output path directory is ordered by the harness name followed by the model name. Here, `native` is the ReAct-loop-based harness implemented by Agency and `gpt-5` is the model. Change either one and the next run writes to a new directory.

## Coming soon

- **Trace replay.** Drive a benchmark against a mock Model API that replays recorded responses, so
  model-side variance drops out of the numbers.
- **External harnesses.** Run the same benchmark under Claude Code, Codex, Grok or OpenCode instead of
  Agency's own ReAct loop.
- **Local GPU.** Run `semantic_embedding`'s encoding on a local GPU. The code is in place behind an
  opt-in; see [that benchmark's README](targeted/semantic_embedding/README.md#under-development-local-gpu).
- **A contributing guide.** The layout a new benchmark has to follow, and how to submit one. Until it
  lands, copy `targeted/paper_summarization/` and open an issue.

## Documentation

- [targeted/](targeted/README.md) — choosing a benchmark, shared setup, common flags
