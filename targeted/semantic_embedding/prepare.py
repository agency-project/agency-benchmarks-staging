"""One-time setup for the semantic embedding benchmark.

Three idempotent steps, in dependency order:

    --fetch   pin the corpus, materialize it, download the encoder weights
    --build   build the sandbox image that bakes both in
    --check   assert the two things that would otherwise fail silently

Run with no flags to do all three. Re-running is cheap: every step skips work that
is already done.

None of this belongs in a measured run. A dataset download or an image build that
happened lazily inside agprof.session() would be charged to the workload being
measured, which is the reason this script exists at all.

Run:
    python prepare.py
    python prepare.py --fetch --force     # reselect the corpus from scratch
    python prepare.py --build --gpu       # NOT YET RELEASED: CUDA-capable base
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
SELECTION_FILE = DATASET_DIR / "selection.json"
MODELS_FILE = DATASET_DIR / "models.json"
MATERIALIZED_DIR = DATASET_DIR / "materialized"
MODEL_CACHE_DIR = HERE / ".cache" / "models"

HF_DATASET = "abisee/cnn_dailymail"
HF_CONFIG = "3.0.0"
HF_SPLIT = "test"

SELECTION_SEED = 24
DOCS_PER_VARIANT = 500

# Three corpora, so the best encoder configuration differs between them. With a single
# corpus there is one fixed right answer, and an agent that uses the length signal is
# indistinguishable from one that memorised it.
#
# The bounds are inclusive-exclusive character counts read off the measured length
# histogram of the split (median 3,563 chars, max 11,991), not guessed: `short` docs fit
# inside a 512-token window whole, `long` docs need three or more windows.
VARIANT_CHAR_BOUNDS = {
    "short": (None, 2000),
    "mixed": (None, None),
    "long": (6143, None),
}
VARIANT_NAMES = tuple(VARIANT_CHAR_BOUNDS)

# Half the queries are the article's own highlights. Those are drawn largely from the
# article lead, so they stay reachable under truncation -- which is exactly why they
# cannot be the only query type.
SUMMARY_QUERY_FRACTION = 0.5

# The other half is a passage lifted verbatim from late in the article. Truncation
# structurally cannot match these, so reaching the target requires covering the whole
# document. 350 chars keeps a tail query comparable in length to a highlights block
# (median 290 chars), so the two kinds are not also a length comparison.
TAIL_QUERY_CHARS = 350
# A tail query starts at least this far in when the article allows it. 2,600 chars is
# past token 512 at any plausible ratio for English news (>=3.0 chars/token), and the
# Dockerfile asserts that with the real tokenizer at build time. Short articles are
# shorter than this, so their tail queries stay inside the window on purpose.
TAIL_QUERY_MIN_START_CHAR = 2600

# Per-variant, because the variants differ in difficulty by design: one shared number would
# be trivial for `short` and unreachable for `long`. Each sits strictly between what the
# cheapest configuration reaches and what a mid-cost one reaches, so clearing it always
# takes a real decision and the largest encoder is never required.
#
# Measured recall@1 for the two configurations --check re-measures (only these two: a target
# asserted against anything dearer would drift until only the expensive answer works):
#
#            minilm-l6 truncate@256   bge-small chunk_mean@512
#   short              0.910                  0.956
#   mixed              0.836                  0.948
#   long               0.754                  0.924
TARGET_RECALL = {
    "short": 0.94,
    "mixed": 0.92,
    "long": 0.85,
}

IMAGE_REPO = "agency-bench/semantic-embedding"


def image_tag() -> str:
    """Tag naming the inputs baked into the image.

    Content-addressed rather than fixed, because a fixed tag cannot express that the image
    is out of date: editing sandbox_runner.py or the corpus and running the benchmark
    without rebuilding leaves the host expecting one thing and the container holding
    another, with nothing to reveal it. With the inputs in the tag, a forgotten rebuild
    asks for an image that does not exist.
    """
    digest = hashlib.sha256()
    for path in (SELECTION_FILE, MODELS_FILE, HERE / "sandbox_runner.py"):
        digest.update(path.read_bytes())
    return f"{IMAGE_REPO}:{digest.hexdigest()[:12]}"

# The CPU base and its torch pin are the Dockerfile's ARG defaults, so they have one
# home; only the --gpu override lives here. Neither is `agency-sandbox:latest`: that is a
# locally built, mutable tag with three variants (Dockerfile, .nvidia, .rocm) shipping
# three different torch builds, and CPU seconds differ materially between them. A local
# image also has no upstream digest, so it cannot be pinned across machines. Both bases
# used here are the ones Agency's own images/Dockerfile{,.nvidia} build on.
BASE_IMAGE_GPU = (
    "nvcr.io/nvidia/pytorch:26.05-py3"
    "@sha256:ca73b4795f0d3ae27e9cd81b1b1f1b7fc6c0a129f7d51a359d2326e95af48a3d"
)

# The GPU path is written and its base image is fixed by digest, but it has never been run
# end to end, so no number it produces has been checked against anything. The opt-in keeps
# it usable for whoever has the hardware without letting a first-time user reach an
# unverified measurement by passing one flag. benchmark.py gates --device cuda on the same
# variable, since a CUDA image is useless without it.
GPU_OPT_IN = "BENCH_ENABLE_GPU"


def gpu_unreleased(flag: str) -> int:
    """Refuse *flag* and say why. Returns the exit code, so callers can `return` it."""
    print(
        f"{flag}: the local GPU path is not yet released. It is implemented but still\n"
        f"under development, so its numbers may be unstable. Set {GPU_OPT_IN}=1 to use it.",
        file=sys.stderr,
    )
    return 2

# Weights only, in safetensors form. Skipping the duplicate .bin/.h5/.ot checkpoints and
# the ONNX/OpenVINO exports cuts roughly two thirds of the download.
ENCODER_ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling/*",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab.txt",
]

# Configurations --check re-measures per variant. The cheap one must miss the target and
# the mid-cost one must clear it. Deliberately not the largest encoder: asserting against
# a configuration the agent should never need would let the target drift upward until only
# the expensive answer works.
CHECK_CHEAP_CONFIG = {"model": "minilm-l6", "max_seq_len": 256, "strategy": "truncate"}
CHECK_SUFFICIENT_CONFIG = {"model": "bge-small", "max_seq_len": 512, "strategy": "chunk_mean"}
# Smallest corpus, used for the sandbox measurement check where only the plumbing matters.
CHECK_SMOKE_VARIANT = "short"
# Threads --check pins the encoder to for the quality measurements, matching
# benchmark.py's default so its numbers are comparable with a default run rather than with
# whatever the host happens to offer. The sandbox check deliberately uses a different
# value, so that "honoured its thread count" is a real assertion rather than a coincidence.
CHECK_THREADS = 8

RUNNER_IN_IMAGE = "/opt/bench/sandbox_runner.py"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command and capture its output, for checks that parse it."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _run_streaming(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command with its output going straight to the terminal.

    Used for image builds and multi-GB downloads: with output captured they are
    indistinguishable from a hung process.
    """
    return subprocess.run(cmd, text=True, **kw)


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def parse_report(output: str) -> dict:
    """Pull the runner's JSON report out of its output.

    The runner prints one object on stdout, but exec() returns stdout and stderr
    together and torch is free to warn into it, so the report is located rather than
    assumed to be the whole thing.
    """
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"no JSON report in runner output: {output[-500:]}")


def load_choices() -> dict:
    """Read the pinned choice space: which encoders exist and which strategies."""
    if not MODELS_FILE.is_file():
        sys.exit(f"missing {MODELS_FILE}")
    return json.loads(MODELS_FILE.read_text())


def load_encoders() -> dict[str, dict]:
    return load_choices()["encoders"]


def load_selection() -> dict:
    if not SELECTION_FILE.is_file():
        sys.exit(f"No pinned corpus at {SELECTION_FILE}. Run: python prepare.py --fetch")
    return json.loads(SELECTION_FILE.read_text())


def _variant_dir(variant: str) -> Path:
    return MATERIALIZED_DIR / variant


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _materialized_ok(variant: str, expected: int) -> bool:
    """True when both files exist at full length.

    Line counts rather than content hashes: an interrupted write is the realistic
    failure, and it always shows up as a short file.
    """
    d = _variant_dir(variant)
    return (
        _line_count(d / "corpus.jsonl") == expected
        and _line_count(d / "queries.jsonl") == expected
    )


# --------------------------------------------------------------------------
# --fetch
# --------------------------------------------------------------------------
def _load_split():
    """Load the CNN/DailyMail split, failing loudly if its schema moved."""
    from datasets import load_dataset

    print(f"Loading {HF_DATASET} ({HF_CONFIG}) split={HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    required = {"id", "article", "highlights"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(
            f"{HF_DATASET} is missing column(s) {sorted(missing)}.\n"
            f"  columns present: {ds.column_names}"
        )
    print(f"  {len(ds)} articles")
    return ds


def _select_variants(lengths: list[int]) -> dict[str, list[int]]:
    """Pick DOCS_PER_VARIANT row indices per variant, bucketed by article length.

    Seeded per variant so the three samples are independent but reproducible. The
    variants may overlap (a long article is also eligible for `mixed`); they are
    separate corpora, so sharing a document between them costs nothing.
    """
    out: dict[str, list[int]] = {}
    for variant, (lo, hi) in VARIANT_CHAR_BOUNDS.items():
        eligible = [
            i
            for i, n in enumerate(lengths)
            if (lo is None or n >= lo) and (hi is None or n < hi)
        ]
        if len(eligible) < DOCS_PER_VARIANT:
            sys.exit(
                f"variant {variant!r}: only {len(eligible)} articles in "
                f"[{lo}, {hi}) chars, need {DOCS_PER_VARIANT}"
            )
        rng = random.Random(f"{SELECTION_SEED}:{variant}")
        picked = rng.sample(eligible, DOCS_PER_VARIANT)
        out[variant] = sorted(picked)
        print(f"  {variant:6s} {len(eligible):6d} eligible -> {DOCS_PER_VARIANT} selected")
    return out


def _tail_span(article: str, seed: str) -> tuple[int, int]:
    """Character span of a passage sampled from late in *article*.

    Seeded per document rather than drawing from a shared stream, so a span depends only on
    its own document. One shared stream makes every span depend on how many draws preceded
    it, so reordering documents, changing the summary/tail split, or adding a draw anywhere
    silently repins the whole corpus -- and the numbers the targets were calibrated against
    then describe a dataset nobody can identify.

    Snapped outward to whitespace so the query is not a mid-word fragment, which would make
    it a tokenizer test rather than a retrieval one.
    """
    rng = random.Random(seed)
    n = len(article)
    length = min(TAIL_QUERY_CHARS, n)
    latest = max(0, n - length)
    earliest = min(max(n // 2, TAIL_QUERY_MIN_START_CHAR), latest)
    start = rng.randint(earliest, latest)
    while start < latest and not article[start].isspace():
        start += 1
    end = min(n, start + length)
    while end < n and not article[end].isspace():
        end += 1
    return start, end


def _build_selection(ds) -> dict:
    """Choose the documents and the query for each, and record it all."""
    lengths = [len(a) for a in ds["article"]]
    ids = ds["id"]
    chosen = _select_variants(lengths)

    variants: dict[str, list[dict]] = {}
    for variant, rows in chosen.items():
        rng = random.Random(f"{SELECTION_SEED}:{variant}:queries")
        n_summary = int(round(DOCS_PER_VARIANT * SUMMARY_QUERY_FRACTION))
        kinds = ["summary"] * n_summary + ["tail"] * (DOCS_PER_VARIANT - n_summary)
        rng.shuffle(kinds)

        entries = []
        for row, kind in zip(rows, kinds):
            entry = {"doc_id": ids[row], "kind": kind}
            if kind == "tail":
                start, end = _tail_span(
                    ds[row]["article"], f"{SELECTION_SEED}:{variant}:{ids[row]}"
                )
                entry["start_char"] = start
                entry["end_char"] = end
            entries.append(entry)
        variants[variant] = entries

    return {
        "dataset": {"name": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT},
        "seed": SELECTION_SEED,
        "docs_per_variant": DOCS_PER_VARIANT,
        "variant_char_bounds": {k: list(v) for k, v in VARIANT_CHAR_BOUNDS.items()},
        "tail_query_chars": TAIL_QUERY_CHARS,
        "tail_query_min_start_char": TAIL_QUERY_MIN_START_CHAR,
        # Carried here rather than imported from prepare.py so that running the
        # benchmark has no import dependency on the setup script.
        "target_recall": TARGET_RECALL,
        "variants": variants,
    }


def _materialize(ds, selection: dict, force: bool) -> None:
    """Write the corpus and query files the image bakes in."""
    by_id = {doc_id: row for row, doc_id in enumerate(ds["id"])}
    expected = selection["docs_per_variant"]

    for variant, entries in selection["variants"].items():
        if _materialized_ok(variant, expected) and not force:
            print(f"  {variant:6s} already materialized")
            continue
        out = _variant_dir(variant)
        out.mkdir(parents=True, exist_ok=True)

        with (out / "corpus.jsonl").open("w") as cf, (out / "queries.jsonl").open("w") as qf:
            for entry in entries:
                row = by_id.get(entry["doc_id"])
                if row is None:
                    sys.exit(
                        f"pinned doc_id {entry['doc_id']} is no longer in "
                        f"{HF_DATASET}; the upstream split changed"
                    )
                record = ds[row]
                cf.write(json.dumps({"doc_id": entry["doc_id"], "text": record["article"]}) + "\n")
                if entry["kind"] == "summary":
                    text = " ".join(record["highlights"].split())
                else:
                    text = " ".join(
                        record["article"][entry["start_char"] : entry["end_char"]].split()
                    )
                qf.write(
                    json.dumps(
                        {"doc_id": entry["doc_id"], "kind": entry["kind"], "text": text}
                    )
                    + "\n"
                )
        print(f"  {variant:6s} wrote {expected} docs + {expected} queries -> {out}")


def _download_encoders(force: bool) -> None:
    """Fetch the encoder weights the image bakes in.

    Pinned by revision commit sha, which is immutable, so the download is
    self-verifying and no separate checksum table is needed.
    """
    from huggingface_hub import snapshot_download

    for key, spec in load_encoders().items():
        dest = MODEL_CACHE_DIR / key
        if (dest / "model.safetensors").is_file() and not force:
            print(f"  {key:10s} already cached")
            continue
        print(f"  {key:10s} {spec['repo']}@{spec['revision'][:12]} ...")
        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            allow_patterns=ENCODER_ALLOW_PATTERNS,
            local_dir=str(dest),
        )
        for required in ("model.safetensors", "1_Pooling/config.json"):
            if not (dest / required).is_file():
                sys.exit(f"{key}: {required} missing after download")

    total = sum(f.stat().st_size for f in MODEL_CACHE_DIR.rglob("*") if f.is_file())
    print(f"  encoder weights: {total / 1e9:.2f} GB in {MODEL_CACHE_DIR}")


def cmd_fetch(args: argparse.Namespace) -> int:
    print("Fetching corpus and encoder weights\n")
    have_selection = SELECTION_FILE.is_file() and not args.force
    need_materialize = args.force or not all(
        _materialized_ok(v, DOCS_PER_VARIANT) for v in VARIANT_NAMES
    )

    if have_selection and not need_materialize:
        print("  corpus already pinned and materialized")
    else:
        ds = _load_split()
        if have_selection:
            selection = load_selection()
            print(f"  reusing pinned selection from {SELECTION_FILE.name}")
        else:
            selection = _build_selection(ds)
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            SELECTION_FILE.write_text(json.dumps(selection, indent=2) + "\n")
            print(f"  wrote {SELECTION_FILE}")
        _materialize(ds, selection, args.force)

    print()
    _download_encoders(args.force)
    return 0


# --------------------------------------------------------------------------
# --build
# --------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> int:
    print("Building the sandbox image\n")
    missing = [v for v in VARIANT_NAMES if not _materialized_ok(v, DOCS_PER_VARIANT)]
    if missing:
        sys.exit(f"corpus not materialized for {missing}. Run: python prepare.py --fetch")
    if not (MODEL_CACHE_DIR / "bge-large" / "model.safetensors").is_file():
        sys.exit("encoder weights missing. Run: python prepare.py --fetch")

    cmd = ["docker", "build", "-t", image_tag()]
    if args.gpu:
        # Empty TORCH_INSTALL leaves the CUDA base's own torch in place; see the
        # Dockerfile for why replacing it would quietly change the measurement.
        cmd += ["--build-arg", f"BASE_IMAGE={BASE_IMAGE_GPU}", "--build-arg", "TORCH_INSTALL="]
        print(f"  base:  {BASE_IMAGE_GPU}")
        print("  torch: (already in the base image)")
    else:
        print("  base:  the Dockerfile's pinned CPU default")
    print(f"  tag:   {image_tag()}\n")

    if args.force:
        cmd.append("--no-cache")
    cmd += ["-f", str(HERE / "Dockerfile"), str(HERE)]

    # No staleness tracking of our own: a changed corpus or runner busts the COPY
    # layer, so Docker's cache makes this a no-op when nothing changed and a rebuild
    # when something did.
    if _run_streaming(cmd, cwd=HERE).returncode != 0:
        sys.exit("docker build failed")

    size = _run(["docker", "image", "inspect", image_tag(), "--format", "{{.Size}}"])
    if size.returncode == 0:
        print(f"\n  built {image_tag()} ({int(size.stdout.strip()) / 1e9:.1f} GB)")
    return 0


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------
def _runner_in_container(config: dict, variant: str, extra: list[str]) -> dict:
    """Run the pinned runner in a throwaway container and return its report.

    Named so a timeout can clean up after itself: `--rm` is a client-side action, so killing
    the client on TimeoutExpired would otherwise leave the container running and burning CPU
    alongside the checks that follow it.
    """
    name = f"agency-bench-check-{os.getpid()}-{variant}-{config['model']}"
    cmd = [
        "docker", "run", "--rm", "--name", name, image_tag(),
        "python", RUNNER_IN_IMAGE,
        "--mode", "embed",
        "--variant", variant,
        "--model", config["model"],
        "--max-seq-len", str(config["max_seq_len"]),
        "--strategy", config["strategy"],
        "--threads", str(CHECK_THREADS),
        *extra,
    ]
    try:
        p = _run(cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        _run(["docker", "rm", "-f", name], timeout=120)
        raise
    if p.returncode != 0:
        raise RuntimeError(f"runner failed (rc={p.returncode}): {p.stderr[-2000:]}")
    return parse_report(p.stdout)


def _describe(config: dict) -> str:
    return f"{config['model']}/{config['strategy']}@{config['max_seq_len']}"


def _check_targets_discriminate() -> bool:
    """The gate: every variant's target must take a real decision to reach.

    Without this a variant can look healthy while being degenerate in either direction.
    A target the cheapest configuration already clears leaves no loop to measure, and one
    nothing reachable clears makes every run a failure -- and both produce traces that
    look perfectly normal.
    """
    print("1. Quality targets discriminate (full corpus, every variant)")
    if _run(["docker", "image", "inspect", image_tag()]).returncode != 0:
        _fail(f"{image_tag()} not built. Run: python prepare.py --build")
        return False

    ok = True
    for variant in VARIANT_NAMES:
        target = TARGET_RECALL[variant]
        try:
            cheap = _runner_in_container(CHECK_CHEAP_CONFIG, variant, [])
            sufficient = _runner_in_container(CHECK_SUFFICIENT_CONFIG, variant, [])
        except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            _fail(f"{variant}: could not run the encoder in the image: {e}")
            ok = False
            continue

        lo, hi = cheap["recall_at_1"], sufficient["recall_at_1"]
        for label, report in (("cheap", cheap), ("sufficient", sufficient)):
            kind = report["recall_at_1_by_kind"]
            print(
                f"     {variant:6s} {label:10s} {_describe(report):26s} "
                f"recall@1 {report['recall_at_1']:.3f}  "
                f"(summary {kind['summary']:.3f}, tail {kind['tail']:.3f})"
            )
        if lo < target <= hi:
            _ok(f"{variant}: {lo:.3f} < target {target} <= {hi:.3f}")
        else:
            _fail(
                f"{variant}: target {target} is not between {lo:.3f} and {hi:.3f}, so "
                "the variant has no gradient. Retune TARGET_RECALL or the query design."
            )
            ok = False
    return ok


def _check_sandbox_measurement() -> bool:
    """The two ways the CPU figures can be quietly wrong.

    The profiler must charge the encoding to the sandbox's own cgroup: sandbox_metrics is
    the headline metric, and if that lookup produced nothing every run would report zero
    CPU and merely look fast. And the encoder must honour the thread count it was given --
    torch otherwise sizes its pool from the host's core count, and the figures describe
    contention instead of encoding.

    Whether the container also got a CPU quota is reported rather than asserted: Agency
    gates that on a cgroup v1 probe that fails on every cgroup v2 host, so on the hosts
    where the profiler works there is usually no quota and thread pinning is what binds.
    """
    threads = 4
    print(f"\n2. Sandbox measurement is sound (--threads {threads})")
    import tempfile

    from agency import agSandbox, agprof
    from agency.agconfig import agConfig
    from agency.agresources import agResourcePoolConfig
    from agency.agsandbox import agSandboxConfig

    tmp = tempfile.TemporaryDirectory(prefix="agency-bench-check-")
    out_dir = Path(tmp.name)
    cfg = agConfig(
        agSandboxConfig(base_image=image_tag()), agResourcePoolConfig(idle_cpus=float(threads))
    )
    sandbox = agSandbox("prepare_check", agconfig=cfg)
    try:
        with agprof.session(out_dir):
            output, rc = sandbox.exec(
                f"python {RUNNER_IN_IMAGE} --mode embed --variant {CHECK_SMOKE_VARIANT} "
                f"--model minilm-l6 --max-seq-len 256 --strategy truncate "
                f"--threads {threads} --max-docs 20",
                timeout=900,
            )
        if rc != 0:
            _fail(f"runner failed inside the sandbox: {output[-1000:]}")
            return False

        report = parse_report(output)
        summary = json.loads((out_dir / "summary.json").read_text())
        rows = summary.get("sandbox_metrics") or []
        cpu = rows[0].get("cpu_time_seconds") if rows else None

        ok = True
        if not cpu:
            _fail(f"sandbox_metrics has no CPU time (rows={len(rows)})")
            ok = False
        else:
            _ok(f"sandbox cgroup CPU time recorded: {cpu:.2f} s")
        if report["threads"] != threads:
            _fail(
                f"encoder ran with {report['threads']} threads, not the {threads} it was "
                "given; CPU figures would measure contention"
            )
            ok = False
        else:
            _ok(f"encoder honoured its thread count: {report['threads']}")

        quota = report.get("cgroup_cpus")
        if quota is None:
            print(
                "     note: the container has no CPU quota, so thread count is the only\n"
                "           bound on parallelism. Agency gates --cpus on a cgroup v1 probe\n"
                "           (container.py:1158) that fails on this cgroup v2 host."
            )
        else:
            _ok(f"container CPU quota applied: {quota}")
        return ok
    finally:
        try:
            sandbox.destroy()
        except Exception as e:  # teardown must not mask the check's own result
            print(f"     warning: sandbox teardown failed: {e}")
        tmp.cleanup()


def cmd_check(args: argparse.Namespace) -> int:
    print("Preflight checks\n")
    results = [_check_targets_discriminate(), _check_sandbox_measurement()]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fetch", action="store_true", help="pin the corpus and download encoders")
    p.add_argument("--build", action="store_true", help="build the sandbox image")
    p.add_argument("--check", action="store_true", help="run the preflight checks")
    p.add_argument(
        "--gpu",
        action="store_true",
        help="NOT YET RELEASED: build on a CUDA-capable base so --device cuda works "
        f"(--build only). Implemented but never run; set {GPU_OPT_IN}=1 to try it.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="redo work that would otherwise be skipped: reselect the corpus, "
        "re-download encoders, rebuild the image without cache",
    )
    args = p.parse_args()

    if args.gpu and not os.environ.get(GPU_OPT_IN):
        return gpu_unreleased("--gpu")

    steps = [
        (args.fetch, cmd_fetch),
        (args.build, cmd_build),
        (args.check, cmd_check),
    ]
    if not any(selected for selected, _ in steps):
        steps = [(True, fn) for _, fn in steps]

    for selected, fn in steps:
        if not selected:
            continue
        rc = fn(args)
        if rc != 0:
            return rc
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
