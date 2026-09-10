# Paper summarization

**Resource: concurrency** · **Dataset: TREC-COVID**

A research topic is answered by a group of `W` agents running at the same time, one per candidate paper.

```
              topic
                |
            [  lead  ]                     forks W summarisers + 1 reducer
                |
    .-----------+-----------+- - - - - - -.
    |           |           |             |
 [ sum ]     [ sum ]     [ sum ]  . . . [ sum ]    W agents, all at the same time
    |           |           |             |
 +-------+   +-------+   +-------+     +-------+
 |sandbox|   |sandbox|   |sandbox|     |sandbox|   one agSandbox each
 |doc_00 |   |doc_01 |   |doc_02 |     |doc_W-1|   one document, mounted read-only,
 +-------+   +-------+   +-------+     +-------+   read with a real `read` tool call
    |           |           |             |
    '-----------+-----------+- - - - - - -'
                |  join
            [ reducer ]                    summaries in, one short report out
                |
              report
```

## Setup

[../README.md](../README.md) covers the venv, the API key and the shared flags. This benchmark also
needs:

- **The `agency-sandbox:latest` image**, built from your Agency checkout (`./images/build.sh`). It is
  never published, so there is nothing to pull.
- **Network access to Hugging Face** during `prepare.py`, for three BeIR files (~110 MB).

```bash
python prepare.py   # choose the topics, download the corpus, record the image
```

Each of the four steps skips if it's already done beofre. `prepare.py --check` is the one to re-run after changing anything: it validates the fixed commits, the corpus digest, a live read-only mount, and the model credentials.

## Run

```bash
python benchmark.py --topics 1 --widths 1  # cheapest episode: 1 episode, 2 agent runs
# OR
python benchmark.py                        # default sweep: widths 1,2,4,8 on one topic, 19 runs
```

## Results

```
output/<harness>/<model>/<qid>/W<width>/t<trial>/
  profiler/*.pt.trace.json   # one lane per agent, under an episode[WN] span
  profiler/summary.{json,md}
  result.json                # the metrics below, plus the full run config
  summaries.json             # what each agent returned, and the reducer's report
  agent_logs/                # <agname>_history.jsonl and _timeline.jsonl per agent
```

One **episode** is one topic at one fan-out width.

```
                        |<-------- fanout_ms ------->|<- reduce_ms ->|
                        |<--------------- wall_ms -------------------->|
                        |                            |               |
   warm-up              |  W x ag.run(summarise)     |  reducer      |   teardown
   fork W+1 agents,     |    -> W pending agdata     |  .run(compile |   destroy
   start a sandbox      |  join: dependency|barrier  |    _report)   |   W+1 sandboxes
   each, mount one doc  |                            |               |
  ---------------------[o]--------------------------[o]-------------[o]---------------
                  fanout_start                 fanout_end      reduce_end

   |<------------------------------ wall_clock_s ------------------------------->|
```

| Metric | Meaning |
|---|---|
| `peak_overlap` | How many of the `W` summarisers were actually running at the same time. `peak_overlap == W` means the group really did overlap; less than `W` means something serialised it |
| `fanout_ms` | How long the `W` summarisers took, from dispatching the first to the last one finishing |
| `reduce_ms` | How long the single reducer took |
| `wall_ms` | `fanout_ms + reduce_ms` |
| `item_ms_p50` | Median per-item latency |
| `makespan_ratio` | `fanout_ms` ÷ the slowest item's latency; `1.0` is the best a fan-out can do |
| `grounding_rate` | Share of evidence quotes found verbatim in the document; `null` when nothing was scorable |
| `throughput_gain` | `W × item_ms_p50(W=1) / fanout_ms(W)`, in `results.md`'s `## By width` table.|

## Parameters

### `benchmark.py`

| Flag | Default | Purpose |
|---|---|---|
| `--widths` | `1,2,4,8` | Group sizes to run, one after another, always ascending however you list them. Each must be `1` to `8`, `8` being the candidate documents recorded per topic. **Keep `1`** since it is the baseline `throughput_gain` divides by |
| `--topics N` | `1` | How many of the recorded topics to run, in qid *string* order |
| `--topic QID` | — | One specific topic id, repeatable. Replaces `--topics` rather than narrowing it |
| `--trials N` | `1` | Repeat each (topic, width). The only way to bound baseline noise |
| `--join-mode` | `dependency` | How the lead waits for the group. `dependency` reads the `W` results one at a time in order; `barrier` waits for all of them at once with `agsync()`. Either way it waits for every summariser — what changes is the shape in the trace |
| `--max-steps` | `24` | Most ReAct turns one agent may take, where each turn is one model call. |
| `--base-image` / `--corpus` | `agency-sandbox:latest` / `.cache/corpus/trec-covid.test` | The image every agent's sandbox starts from; the corpus directory `prepare.py --corpus` wrote |

Also reads `BENCH_WIDTHS`, `BENCH_TOPICS`, `BENCH_JOIN_MODE`, `BENCH_BASE_IMAGE` and `BENCH_CORPUS`.
`--topics 8` gives qids 1, 10, 11, … not 1 through 8, because a BeIR query id is a string.

### `prepare.py`

| Flag | Purpose |
|---|---|
| *(none)* | Run all four steps in order |
| `--dataset` | Select and record the topics and their candidate draw |
| `--corpus` | Write out the selected documents, verified against their checksums |
| `--image` | Record the sandbox base image's content id in the manifest |
| `--check` | Check the selection, the corpus, a live mount and the model before a run |
| `--force` | Redo the selection or the corpus extraction |
| `--topics` (`8`) / `--candidates` (`8`) / `--seed` (`24`) | The selection; needs `--dataset --force`. `--candidates` sets the maximum sweepable width |
| `--base-image` (`agency-sandbox:latest`) | The image `--image` records and `--check` verifies |
| `--corpus-dir` | `benchmark.py`'s `--corpus` under another name, since here `--corpus` selects the step |
| `--backend` / `--model` / `--region` / `--context-limit` | Which model `--check` probes |

## Cleaning up

A run that finishes destroys its own sandboxes. Use this only after killing a run half-way through where necessary:

```bash
RT=$(basename "$(command -v podman || command -v docker)")   # whichever Agency resolved
$RT ps -a --filter 'name=^sandbox-' --format '{{.Names}} {{.Status}}'   # look first
$RT ps -aq --filter 'name=^sandbox-' | xargs -r $RT rm -f    # any sandbox left behind
rm -rf output/* .cache                                       # traces, corpus, downloads
```
