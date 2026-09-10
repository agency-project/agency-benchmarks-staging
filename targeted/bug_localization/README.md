# Bug localization

**Resource: storage I/O** · **Dataset: SWE-bench Verified**

An agent is given a real GitHub issue and a real repository at its buggy commit, and must report which
source files need to change. This involves reading through a large unfamiliar tree, which is the disk I/O being measured.

```
   sync + drop the host page cache          so every read has to reach the disk
              |
  +-----------v--------------------------------------------------+
  |  sandbox, built from this instance's SWE-bench image         |
  |                                                              |
  |   GitHub issue ---> [ bug_localizer ]                        |
  |   (never names          |  grep  |                           |
  |    the files)           |  glob  +---> /testbed              |
  |                         |  read  |     the repository at     |
  |                         '--------'     its buggy commit      |
  |                     ReAct loop, capped by --max-steps        |
  +----------------------------|---------------------------------+
                               |  the sandbox's cgroup counts
                               v  every block-device read
                          io_read_mb
```

The files it reports are scored against the gold patch as a sanity check, while `io_read_mb` and
the trace are the result.

## Setup

[../README.md](../README.md) covers the venv, the API key and the shared flags. This benchmark also
needs:

- **About 25 GB of disk** for the images, and your user in the `docker` group.
- **Passwordless `sudo`** for the page-cache drop. Without it the run still completes, but warns and
  reports `io_read_mb: 0`.
- **Network access** to Hugging Face (the dataset), Docker Hub (the images) and GitHub (`ripgrep`).

```bash
python prepare.py   # fetch ripgrep, choose the instances, pull the bases, build, then check
```

Each of the five steps skips if already done. The first run downloads about 25 GB and takes a while.

> **By default, the host page cache is dropped before every instance.** Disk-read numbers are always
> zero otherwise. This is host-wide: every process on the machine slows briefly while caches refill.
> Nothing is lost since only clean pages are dropped, and `sync` runs first. On a shared machine,
> please be careful. You can also use `--no-drop-caches` and only analyse the trace.

## Run

```bash
python benchmark.py --limit 1   # just the first instance
# OR pick one
python benchmark.py --instance django__django-15629
# OR run all
python benchmark.py             # all 12
```

Instances run one at a time, each its own agent in its own sandbox, each a ReAct loop invoking one model
call per turn up to `--max-steps`.

## Results

```
output/<harness>/<model>/<instance_id>/
  profiler/*.pt.trace.json   # one lane for the agent, with a tool:* span per grep/glob/read
  profiler/summary.{json,md}
  result.json                # the metrics below, plus the full run config
  agent_logs/                # <agname>_history.jsonl and _timeline.jsonl
```

`results.json` and `results.md` sit one level up, aggregated with a per-difficulty breakdown.

| Metric | Meaning |
|---|---|
| `io_read_mb` | Megabytes the sandbox actually read from disk, taken from its cgroup. **The primary result** |
| `precision` / `recall` / `f1` | Reported files against the files the gold patch touches. A sanity check, not the result |
| `status` | `ok`, `max_steps_exceeded`, `error` or `exception` |
| `wall_clock_s` | End to end for the instance, including sandbox start |

## Parameters

### `benchmark.py`

| Flag | Default | Purpose |
|---|---|---|
| `--instance ID` | all 12 | One instance id from `dataset/selected_instances.jsonl`; repeat the flag to select several |
| `--limit N` | all 12 | The first N in file order, which is grouped by difficulty: three `1-4 hours`, then `15 min - 1 hour`, `<15 min`, `>4 hours`. Applied after `--instance` |
| `--max-steps` | `60` | Most ReAct turns the agent may take, where each turn is one model call |
| `--drop-caches` / `--no-drop-caches` | **on** | Drop the host page cache before each instance. Turning it off makes `io_read_mb` read 0 |

### `prepare.py`

| Flag | Purpose |
|---|---|
| *(none)* | Run all five steps in order |
| `--rg` | Fetch the fixed `ripgrep` version the images bake in |
| `--dataset` | Select the instances and fetch their issue text |
| `--pull` | Pull the SWE-bench base images, each named by digest |
| `--build` | Build the per-instance sandbox images |
| `--check` | Check the repository path, the profiler and the model before a run |
| `--force` | Reselect the instances, rebuild images, re-download `ripgrep` |
| `--seed` (`24`) / `--min-repo-mb` (`100`) | The selection; needs `--dataset --force` |
| `--backend` / `--model` / `--region` / `--context-limit` | Which model `--check` probes |

The dataset is 12 SWE-bench Verified instances, 3 from each of its 4 difficulty buckets, chosen under a
fixed seed, preferring repositories of at least 100 MB. `dataset/selected_instances.jsonl` records the choice and is committed; it holds identifiers only. The issue text and gold patches are upstream content, so `prepare.py --dataset` fetches them into `dataset/materialized/` instead of committing them.

## Cleaning up

```bash
docker image rm $(docker images 'agency-bench/bug-localization' -q)   # derived images
docker image prune                                                    # unreferenced base layers
rm -rf output/* .cache                                                # traces and the ripgrep cache
```
