"""Corpus encoding and retrieval scoring. Runs INSIDE the sandbox image.

This is the work the benchmark measures, and it is a separate file for one reason:
it has to execute in the container, where the encoder weights, the corpus and a
pinned torch live. benchmark.py runs in the driver process on the host, whose only
channel into the container is a shell command; encoding there instead would charge
the CPU to the driver and leave sandbox_metrics, the headline metric, empty.
It is the same role ripgrep plays in the bug_localization benchmark, and it is baked
into the image the same way -- nothing is written into a live container.

The parameters come from the agent, the code does not: identical parameters produce
identical kernels, which is what makes two runs comparable.

Writes exactly one JSON object to stdout. Never imported by the host.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/opt/bench/data"))
MODELS_DIR = Path(os.environ.get("BENCH_MODELS_DIR", "/opt/models"))
ENCODERS_FILE = Path(os.environ.get("BENCH_ENCODERS_FILE", "/opt/bench/models.json"))

# Enough overlap that a sentence straddling a window boundary survives in one piece,
# small enough not to inflate the sequence count much. Not agent-tunable: it trades
# nothing interesting against cost.
CHUNK_OVERLAP_TOKENS = 64


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def cgroup_cpu_quota() -> "int | None":
    """CPUs this container is limited to, or None when it has no limit.

    nproc reports the host's core count even under `docker run --cpus=N`, so the cgroup
    is the only honest source. Returns None rather than a guess so a caller can tell
    "limited to 8" apart from "unlimited on a 32-core host" -- the report records which
    it was, because the two produce very different CPU figures.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return None


def pin_threads(threads: int) -> int:
    """Fix how many threads the encoder may use.

    Left alone, torch sizes its pool from the host's core count -- 16 threads on a
    32-core host regardless of what this container is entitled to -- and the resulting
    figure describes thread contention rather than the encoder. This is also the only
    CPU bound that reliably holds, since a container quota depends on the runtime
    applying one. Set before torch is imported so the pool is built at the right size.
    """
    # RAYON_NUM_THREADS is not redundant: the Rust tokenizer runs on its own rayon pool
    # and ignores the OMP variables entirely. Measured on the mixed corpus, tokenizing
    # unpinned draws 12.9 CPUs against a budget of 8. It is only a second or so of CPU
    # against an encode measured in minutes, but it is enough to put the sampled peak
    # far above the budget and to blunt a thread-scaling sweep.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[var] = str(threads)
    return threads


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_choices() -> dict:
    return json.loads(ENCODERS_FILE.read_text())


def load_encoders() -> dict[str, dict]:
    return load_choices()["encoders"]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(variant: str, max_docs: int | None) -> list[dict]:
    docs = _read_jsonl(DATA_DIR / variant / "corpus.jsonl")
    return docs[:max_docs] if max_docs else docs


def load_queries(variant: str, doc_ids: set[str]) -> list[dict]:
    """Queries whose gold document is in the corpus.

    Filtered rather than assumed aligned, so --max-docs cannot leave queries pointing
    at documents that were not encoded -- those would score as misses and quietly
    depress recall.
    """
    return [q for q in _read_jsonl(DATA_DIR / variant / "queries.jsonl") if q["doc_id"] in doc_ids]


def load_encoder(spec: dict, key: str, device: str, max_seq_len: int):
    # Checked before the load, not after: the weights are up to 1.3 GB.
    if max_seq_len > spec["max_seq_len"]:
        raise SystemExit(
            f"max_seq_len {max_seq_len} exceeds {key}'s trained limit of {spec['max_seq_len']}"
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(MODELS_DIR / key), device=device)
    model.max_seq_length = max_seq_len
    return model


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------
def chunk_starts(n_tokens: int, window: int, overlap: int) -> range:
    """Window start offsets for a sequence of *n_tokens*.

    The single authority for how a document is split, so the cost that --mode stats
    predicts and the cost --mode embed actually pays cannot drift apart.

    Two bounds here are load-bearing, and both were wrong in an earlier version:

    * The overlap is capped at a quarter of the window. A fixed 64-token overlap can meet
      or exceed a small window -- at max_seq_len=64 the window is 62, the step collapses
      to one token, and a 900-token document becomes 900 chunks instead of 15. The agent
      may legally ask for that, so it cannot be left to clamp.
    * Starts stop `overlap` short of the end, because the previous window already reaches
      it. Running to `n_tokens` instead emits a redundant tail: at window 510, a
      900-token document gets a third chunk of 8 tokens lying entirely inside the second,
      which both overstates cost and -- since chunk_mean averages chunk vectors with
      equal weight -- drags the document's embedding toward an 8-token fragment.

    The same effective overlap is used for both, which is what guarantees the last window
    still reaches the end of the sequence.
    """
    if n_tokens <= window:
        return range(1)
    # Past this point n_tokens > window >= 4 * effective_overlap, so the stop bound is
    # always positive and needs no clamp of its own.
    effective_overlap = min(overlap, window // 4)
    step = window - effective_overlap
    return range(0, n_tokens - effective_overlap, step)


def chunk_token_ids(ids: list[int], window: int, overlap: int) -> list[list[int]]:
    """Split token ids into overlapping windows covering the whole sequence."""
    return [ids[start : start + window] for start in chunk_starts(len(ids), window, overlap)]


def build_sequences(
    model, texts: list[str], strategy: str, max_seq_len: int
) -> tuple[list[str], list[int], int]:
    """Expand documents into the sequences to encode.

    Returns the sequence texts, how many each document contributed (which is what lets
    the vectors be pooled back afterwards), and the exact content-token count. Every
    strategy goes through the same tokenization, so tokenize cost is charged identically
    across configurations and token throughput is measured rather than estimated.
    """
    tokenizer = model.tokenizer
    # Two slots go to [CLS]/[SEP], so windowing on content tokens at the full
    # max_seq_len would silently cost two tokens from every chunk.
    window = max_seq_len - 2
    sequences: list[str] = []
    counts: list[int] = []
    tokens = 0
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        if strategy == "truncate":
            chunks = [ids[:window]]
        else:
            chunks = chunk_token_ids(ids, window, CHUNK_OVERLAP_TOKENS)
        sequences.extend(tokenizer.decode(c, skip_special_tokens=True) for c in chunks)
        counts.append(len(chunks))
        # Counted before the decode/re-encode round trip, which is not exactly lossless:
        # measured on the mixed corpus about 2-3% of chunks re-tokenize a few tokens longer
        # than the window and get truncated, drifting this total by under 0.05%. Uniform
        # across configurations and deterministic for fixed parameters, so comparability
        # holds; it is just not an exact count of tokens the model saw.
        tokens += sum(len(c) for c in chunks)
    return sequences, counts, tokens


# --------------------------------------------------------------------------
# encoding and scoring
# --------------------------------------------------------------------------
def _encode(model, texts: list[str], batch_size: int):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def pool_document_vectors(vectors, counts: list[int], strategy: str):
    """Reduce per-sequence vectors to what the search will compare against.

    chunk_mean collapses each document to one vector; chunk_max keeps every chunk and
    defers the reduction to scoring, which costs memory in exchange for matching on the
    single best-fitting passage.
    """
    import numpy as np

    if strategy == "truncate":
        return vectors, None

    segment_starts = np.cumsum([0] + counts[:-1])
    if strategy == "chunk_mean":
        summed = np.add.reduceat(vectors, segment_starts, axis=0)
        pooled = summed / np.asarray(counts, dtype=vectors.dtype)[:, None]
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-12), None
    return vectors, segment_starts


def retrieve(doc_vectors, segment_starts, query_vectors):
    """Index of the best-matching document for each query.

    Both sides are normalised, so a dot product is cosine similarity.
    """
    import numpy as np

    sim = query_vectors @ doc_vectors.T
    if segment_starts is not None:
        # Chunks are contiguous per document, so a segmented max reduces the
        # per-chunk scores to a per-document score in one pass.
        sim = np.maximum.reduceat(sim, segment_starts, axis=1)
    return np.argmax(sim, axis=1)


def score(predicted, queries: list[dict], doc_ids: list[str]) -> dict:
    """recall@1 overall and split by query kind.

    The split is the diagnostic that matters: a configuration that scores well on
    summary queries and badly on tail queries is not weak, it is truncating.
    """
    by_kind: dict[str, list[int]] = {}
    hits = 0
    for query, doc_index in zip(queries, predicted):
        hit = int(doc_ids[doc_index] == query["doc_id"])
        hits += hit
        by_kind.setdefault(query["kind"], []).append(hit)
    return {
        "recall_at_1": round(hits / len(queries), 4) if queries else 0.0,
        "recall_at_1_by_kind": {
            kind: round(sum(v) / len(v), 4) for kind, v in sorted(by_kind.items())
        },
        "queries": len(queries),
    }


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _rusage() -> dict:
    """Per-attempt CPU and memory.

    The profiler's cgroup figure covers a whole session, so it cannot tell one
    encoding attempt from the next; this process is one-shot, so its own rusage can.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_seconds": round(_cpu_seconds(), 2),
        "peak_rss_mb": round(ru.ru_maxrss / 1024, 1),
    }


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def _mean_chunks(lengths, max_seq_len: int) -> float:
    """Chunks per document a chunking strategy would encode at *max_seq_len*.

    Uses chunk_starts(), so this is the cost the agent would actually pay rather than an
    independent formula that could disagree with the chunker.
    """
    window = max_seq_len - 2
    counts = [len(chunk_starts(int(n), window, CHUNK_OVERLAP_TOKENS)) for n in lengths]
    return round(sum(counts) / len(counts), 2) if counts else 0.0


def cmd_stats(args: argparse.Namespace) -> dict:
    """Token-length distribution of a sample of the corpus.

    Sampled, not exhaustive: tokenizing every document would add a visible slice of CPU
    to the run and blur the attribution of the encoding it is meant to inform. The
    sample is the head of a fixed corpus order, so it is deterministic.
    """
    import numpy as np
    from transformers import AutoTokenizer

    spec = load_encoders()[args.model]
    docs = load_corpus(args.variant, None)
    sample = docs[: args.sample_size]

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / args.model))
    encoded = tokenizer([d["text"] for d in sample], add_special_tokens=False)["input_ids"]
    lengths = np.asarray([len(ids) for ids in encoded])

    return {
        "variant": args.variant,
        "model": args.model,
        "model_max_seq_len": spec["max_seq_len"],
        "corpus_docs": len(docs),
        "sampled_docs": len(sample),
        "tokens_p50": int(np.percentile(lengths, 50)),
        "tokens_p90": int(np.percentile(lengths, 90)),
        "tokens_p95": int(np.percentile(lengths, 95)),
        "tokens_max": int(lengths.max()),
        "tokens_mean": round(float(lengths.mean()), 1),
        "fraction_over_256_tokens": round(float((lengths > 256).mean()), 3),
        "fraction_over_512_tokens": round(float((lengths > 512).mean()), 3),
        "chunks_per_doc_at_256": _mean_chunks(lengths, 256),
        "chunks_per_doc_at_512": _mean_chunks(lengths, 512),
        "wall_s": round(time.perf_counter() - started, 2),
    }


def cmd_embed(args: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(args.threads)

    encoders = load_encoders()
    spec = encoders[args.model]
    docs = load_corpus(args.variant, args.max_docs)
    doc_ids = [d["doc_id"] for d in docs]
    queries = load_queries(args.variant, set(doc_ids))

    t0 = time.perf_counter()
    cpu_before_load = _cpu_seconds()
    model = load_encoder(spec, args.model, args.device, args.max_seq_len)
    load_s = time.perf_counter() - t0
    # Charged separately so a configuration is ranked on encoding rather than on setup.
    # This window covers importing sentence-transformers as well as reading the weights --
    # for the smaller encoders the import dominates -- and it is the part that varies with
    # encoder size, enough to flip a close comparison if it were left in.
    load_cpu_seconds = _cpu_seconds() - cpu_before_load

    with torch.inference_mode():
        t0 = time.perf_counter()
        sequences, counts, doc_tokens = build_sequences(
            model, [d["text"] for d in docs], args.strategy, args.max_seq_len
        )
        tokenize_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        sequence_vectors = _encode(model, sequences, args.batch_size)
        encode_docs_s = time.perf_counter() - t0

        doc_vectors, segment_starts = pool_document_vectors(
            sequence_vectors, counts, args.strategy
        )

        t0 = time.perf_counter()
        query_vectors = _encode(
            model, [spec["query_prefix"] + q["text"] for q in queries], args.batch_size
        )
        encode_queries_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        predicted = retrieve(doc_vectors, segment_starts, query_vectors)
        search_s = time.perf_counter() - t0

    report = {
        "variant": args.variant,
        "model": args.model,
        "max_seq_len": args.max_seq_len,
        "strategy": args.strategy,
        "batch_size": args.batch_size,
        "device": args.device,
        "threads": args.threads,
        # None when the container has no CPU quota, which changes how the CPU figures
        # should be read: threads then bound parallelism, not the cgroup.
        "cgroup_cpus": cgroup_cpu_quota(),
        "corpus_docs": len(docs),
        "sequences_encoded": len(sequences),
        "tokens_encoded": doc_tokens,
        "vectors_per_doc": round(len(sequences) / len(docs), 2) if docs else 0.0,
        "embedding_dim": spec["dim"],
        **score(predicted, queries, doc_ids),
        "load_s": round(load_s, 2),
        "tokenize_s": round(tokenize_s, 2),
        "encode_docs_s": round(encode_docs_s, 2),
        "encode_queries_s": round(encode_queries_s, 2),
        "search_s": round(search_s, 3),
        "load_cpu_seconds": round(load_cpu_seconds, 2),
        **_rusage(),
    }
    # What the cost comparison should use: total process CPU minus the one-off weight
    # load. benchmark.py ranks configurations on this.
    report["encode_cpu_seconds"] = round(report["cpu_seconds"] - load_cpu_seconds, 2)
    # Over document encoding only: queries are short and few, and folding them in
    # would make throughput depend on the query set rather than on the corpus.
    report["sequences_per_s"] = (
        round(len(sequences) / encode_docs_s, 1) if encode_docs_s else 0.0
    )
    report["tokens_per_s"] = round(doc_tokens / encode_docs_s) if encode_docs_s else 0
    return report


# Only these modes do measured work, so only these can run away after a timeout.
WORKLOAD_MODES = ("embed", "stats")


def _runner_pids() -> "list[int]":
    """PIDs of other processes running this script in one of its workload modes.

    Matching on the mode, not merely on the script path, is what keeps this safe. A reap or
    verify invocation -- and the shell wrapping it, which repeats the whole command line --
    must never be a target: Docker's `RUN` wraps the build-time verify in `/bin/sh -c`, and
    an earlier version of this function killed that shell and then reported success.

    Both argv forms have to be checked. /proc/<pid>/cmdline is NUL-separated and the NULs
    survive the decode, so a direct invocation yields exact tokens, while a wrapping shell
    holds the same text with real spaces.
    """
    me = os.getpid()
    pids = []
    for entry in os.listdir("/proc"):
        # PID 1 is skipped as well as our own: the kernel ignores SIGKILL sent to a PID
        # namespace's init from inside that namespace, so including it would report a
        # permanent `remaining` and fail every reap. In the benchmark's real topology PID 1
        # is the container's `tail -f /dev/null` and never a workload process anyway; it
        # only matters for anyone hand-testing with `docker run ... bash -c`.
        if not entry.isdigit() or int(entry) in (me, 1):
            continue
        try:
            cmdline = Path(f"/proc/{entry}/cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:  # the process exited while we were looking at it
            continue
        if "sandbox_runner.py" not in cmdline:
            continue
        argv = cmdline.split("\0")
        if any(mode in argv or f"--mode {mode}" in cmdline for mode in WORKLOAD_MODES):
            pids.append(int(entry))
    return pids


def cmd_reap(_args: argparse.Namespace) -> dict:
    """Kill any other run of this script in this container, and confirm it died.

    exec()'s timeout only stops the host-side `docker exec` client, so a timed-out encode
    keeps running and its CPU is charged to every later attempt in the same sandbox.
    Written in Python against /proc rather than with pkill, because the slim base image
    ships no procps -- and a `pkill || true` would have failed silently, which is the exact
    class of bug this benchmark exists to avoid.

    `remaining` is the point of the second scan: the caller cannot otherwise tell "nothing
    to kill" from "the kill did not work", and those demand opposite responses.

    Safe to kill every workload process only because tool calls are strictly sequential --
    one skill thread, run_in_subprocess=False. Fanning variants out inside one container
    would make this reap another variant's live encode.
    """
    killed = []
    for pid in _runner_pids():
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    time.sleep(0.5)  # SIGKILL is not synchronous; give the kernel a moment to reap
    return {"reaped": killed, "remaining": _runner_pids()}


def cmd_verify(_args: argparse.Namespace) -> dict:
    """Assert the image is internally consistent. Run at build time, not run time.

    Three failure modes this catches, all of which would otherwise surface as a
    plausible-looking but meaningless benchmark result: an encoder that does not load,
    a corpus that does not line up with its queries, and tail queries that do not
    actually fall outside a 512-token window. prepare.py can only approximate that last
    one in characters, since the tokenizer lives here and not on the host.
    """
    from transformers import AutoTokenizer

    selection = json.loads(Path("/opt/bench/selection.json").read_text())
    expected = selection["docs_per_variant"]
    checks: list[str] = []

    for key, spec in load_encoders().items():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(MODELS_DIR / key), device="cpu")
        dim = model.encode(["a", "b"], convert_to_numpy=True).shape[1]
        if dim != spec["dim"]:
            raise SystemExit(f"{key}: encoded dim {dim}, expected {spec['dim']}")
        checks.append(f"{key} loads, dim {dim}")
        del model

    for variant, entries in selection["variants"].items():
        docs = {d["doc_id"]: d["text"] for d in load_corpus(variant, None)}
        queries = _read_jsonl(DATA_DIR / variant / "queries.jsonl")
        if len(docs) != expected or len(queries) != expected:
            raise SystemExit(
                f"{variant}: {len(docs)} docs and {len(queries)} queries, expected {expected}"
            )
        orphans = {q["doc_id"] for q in queries} - set(docs)
        if orphans:
            raise SystemExit(f"{variant}: {len(orphans)} queries name documents not in the corpus")
        checks.append(f"{variant} complete: {expected} docs and queries, all aligned")

        # A 512-token window is the widest any of these encoders offers, so a tail query
        # beyond it cannot be reached by truncation at any setting.
        tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "bge-base"))
        prefixes = [
            docs[e["doc_id"]][: e["start_char"]] for e in entries if e["kind"] == "tail"
        ]
        encoded = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        beyond = sum(1 for ids in encoded if len(ids) > 512)
        if variant == "long" and beyond != len(prefixes):
            raise SystemExit(
                f"long: only {beyond}/{len(prefixes)} tail queries start past token 512; "
                "truncation would reach them and the quality gradient would collapse"
            )
        checks.append(f"{variant} tail queries past token 512: {beyond}/{len(prefixes)}")
        del tokenizer

    # Run as its own process under a shell rather than calling cmd_reap() in-process, so
    # this exercises the shape the benchmark actually uses -- a reap whose own wrapper
    # shares its command line. That wrapper dimension is where the first version of this
    # went wrong, killing its parent and reporting success.
    import subprocess

    proc = subprocess.run(
        [sys.executable, __file__, "--mode", "reap"], capture_output=True, text=True
    )
    reap = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    if proc.returncode != 0 or reap.get("reaped") or reap.get("remaining"):
        raise SystemExit(
            f"--mode reap is not working on a clean container "
            f"(rc={proc.returncode}): {proc.stdout[-300:]} {proc.stderr[-300:]}"
        )
    checks.append("reap returns cleanly on an idle container, invoked as its own process")

    for line in checks:
        print(f"  ok  {line}", file=sys.stderr)
    return {"verified": True, "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("stats", "embed", "verify", "reap"), required=True)
    p.add_argument("--variant", help="required for --mode stats and embed")
    p.add_argument("--model", help="required for --mode stats and embed")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--strategy", choices=tuple(load_choices()["strategies"]), default="truncate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="defaults to the container's own CPU quota, which is what the "
        "measurement is scoped to",
    )
    # Unclamped here on purpose, so the corpus can be characterised in full for
    # documentation. The agent-facing cap is MAX_STATS_SAMPLE in benchmark.py.
    p.add_argument("--sample-size", type=int, default=200, help="--mode stats only")
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="encode only the first N documents; for preflight checks, not for runs",
    )
    args = p.parse_args()

    modes = {"stats": cmd_stats, "embed": cmd_embed, "verify": cmd_verify, "reap": cmd_reap}
    if args.mode in ("stats", "embed"):
        if not (args.variant and args.model):
            sys.exit(f"--mode {args.mode} requires --variant and --model")
        if args.model not in load_encoders():
            sys.exit(f"unknown model {args.model!r}; valid: {', '.join(load_encoders())}")
    args.threads = pin_threads(args.threads or cgroup_cpu_quota() or os.cpu_count() or 1)

    print(json.dumps(modes[args.mode](args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
