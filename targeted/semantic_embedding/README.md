# Semantic embedding

**Resource: CPU compute** · **Dataset: CNN/DailyMail**

An agent is given a corpus of news articles and a retrieval quality target, and must find the encoder
configuration that reaches it **at the lowest CPU cost**. If a smaller encoder or a cheaper strategy also clears the target, the expensive one is the wrong answer.

```
   one corpus variant + a recall@1 target
              |
  +-----------v-------------------------------------------------+
  |  sandbox (one per variant), encoders and code baked in      |
  |                                                             |
  |   [ embedding_tuner ]                                       |
  |        |                                                    |
  |        |--- corpus_stats ------> how long the documents are  |
  |        |                         (cheap, sampled)            |
  |        |                                                    |
  |        |--- embed_corpus(model, max_seq_len, strategy,      |
  |        |                 batch_size)                        |
  |        |         |                                          |
  |        |         v                                          |
  |        |    encode 500 documents with a real transformer,    |
  |        |    then score recall@1        <-- the measured work |
  |        |         |                                          |
  |        '<--------'  recall + CPU cost, up to                 |
  |                     --max-embed-calls attempts               |
  +----------------------------|--------------------------------+
                               |  the sandbox's cgroup counts
                               v  the CPU time it burned
                      encode_cpu_seconds
```

## Setup

[../README.md](../README.md) covers the venv, model access and the shared flags. This benchmark also
needs:

- **Network access** to Hugging Face, PyPI, Docker Hub and GitHub, during `prepare.py` only. A measured
  run never fetches anything — the image sets `HF_HUB_OFFLINE=1`.

```bash
python prepare.py   # download the corpus and encoders, build the image, then check
```

Each of the three steps skips if already done. Budget 25–30 minutes from a clean checkout to a first
result depending on the machine this is running on.

## Run

```bash
python benchmark.py --variant short   # the cheapest variant
# OR run all three, cheapest first
python benchmark.py
```

Variants run one at a time, each its own agent in its own sandbox. One pass over one variant costs
roughly 80 to 2,800 CPU-seconds depending on the configuration the agent picks.

## Results

```
output/<harness>/<llm-model>/<variant>/
  profiler/*.pt.trace.json   # one lane for the agent, with a tool:* span per embed_corpus call
  profiler/summary.{json,md}
  result.json                # every attempt, the metrics below, plus the full run config
  agent_logs/                # <agname>_history.jsonl and _timeline.jsonl
```

`results.json` and `results.md` sit one level up, with a per-attempt table. `<llm-model>` is the model
driving the agent, not the encoder it chose — that is in `result.json`.

| Metric | Meaning |
|---|---|
| `cheapest_passing_encode_cpu_seconds` | CPU time of the cheapest configuration that hit the target. |
| `cheapest_passing_config` | Which configuration that was |
| `reported_config` | What the agent actually chose to ship. Equal to the cheapest passing one when it got it right |
| `tokens_per_s` | Encoding throughput, per attempt. The right axis for a thread sweep |
| `recall_at_1` | Fraction of queries whose nearest neighbour is the right document |
| `max_recall_by_kind` | Recall split by query kind. Strong on `summary` and weak on `tail` means truncation, not a weak encoder |
| `target_met` | Whether any attempt cleared the target |
| `reap_failed` | An encode timed out and could not be confirmed killed, so the CPU total may include unaccounted work. Discard the variant |

CPU-seconds is not comparable across thread counts. For a fair comparison, you should only compare runs  at equal `--threads`. Also, exit codes, in precedence order, refer to: `130` after Ctrl-C, `2` if any variant ended `reap_failed`, `0` if at least one variant finished `ok`, `1` otherwise. Ctrl-C still writes results for the variant in flight.

## What the agent works with

Three corpus variants of 500 documents each, drawn from CNN/DailyMail (`abisee/cnn_dailymail` 3.0.0, test
split) under seed 24 and bucketed by article length:

| Variant | Article chars | Median tokens | Over 512 tokens | Target recall@1 | Best answer |
|---|---|---|---|---|---|
| `short` | < 2,000 | 352 | 0% | 0.94 | widen the window, don't chunk |
| `mixed` | any | 803 | 76% | 0.92 | chunk |
| `long` | ≥ 6,143 | 1,607 | 100% | 0.85 | chunk |

One query per document, the document itself being the right answer. Half the queries are the article's
own `highlights` (`summary` in `max_recall_by_kind`); half quote a passage from late in the article
(`tail`).

The agent chooses from four encoders: `minilm-l6` (23M parameters, 256-token window), `bge-small`
(33M, 512), `bge-base` (109M, 512), `bge-large` (335M, 512); and three strategies: `truncate` (first
window only), `chunk_mean` (overlapping windows averaged into one vector), and `chunk_max` (every chunk
kept, scored on its best match). We standardise the encoding code across configruations, as defined in
[`sandbox_runner.py`](sandbox_runner.py).

## Under development: local GPU

Encoding on a local GPU (`prepare.py --build --gpu`, then `benchmark.py --device cuda`) is provided but
has been under active development, so the performance may be unstable. Both flags will not run unless `BENCH_ENABLE_GPU=1` is set.

## Parameters

### `benchmark.py`

| Flag | Default | Purpose |
|---|---|---|
| `--variant` | all three | `short`, `mixed` or `long` from the table above. Repeat the flag to select several |
| `--limit N` | all three | The first N variants in `short`, `mixed`, `long` order, which is cheapest first. `--limit 1` is `short` |
| `--threads N` | `8` | Threads the encoder may use, which is what bounds the CPU it can draw. Vary this for a scaling sweep, and compare only within one value |
| `--target-recall` | per variant | One recall@1 target for every variant selected, replacing the per-variant targets above |
| `--max-embed-calls` | `6` | Encodings the agent may spend per variant. Its whole budget for searching |
| `--max-steps` | `40` | Most ReAct turns the agent may take, where each turn is one model call |
| `--embed-timeout` | `1800` | Seconds one encoding may take before it is killed. A kill that cannot be confirmed sets `reap_failed` |

### `prepare.py`

| Flag | Purpose |
|---|---|
| *(none)* | Run all three steps in order |
| `--fetch` | Select and write out the corpus, download the encoder weights |
| `--build` | Build the sandbox image |
| `--check` | Verify the quality targets and the profiler's CPU attribution |
| `--force` | Redo work that would otherwise be skipped |

The selection constants live in `prepare.py` and need `--fetch --force` to take effect:
`SELECTION_SEED` (`24`), `DOCS_PER_VARIANT` (`500`), `VARIANT_CHAR_BOUNDS` and `TARGET_RECALL`.

## Cleaning up

```bash
docker image rm $(docker images 'agency-bench/semantic-embedding' -q)
docker image prune
rm -rf output/* .cache dataset/materialized
```
