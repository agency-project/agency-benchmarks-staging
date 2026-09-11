"""
One-time setup for the paper-summarization benchmark.

    python prepare.py             # all steps below, in dependency order

    python prepare.py --dataset   # select and pin the TREC-COVID topics and their candidates
    python prepare.py --corpus    # materialize just those candidates' text, digest-verified
    python prepare.py --image     # record the sandbox base image and its image id
    python prepare.py --check     # verify the pin, the corpus, the mount and the model

Every step is idempotent: re-running skips whatever is already done. Add --force to redo the
selection or the corpus extraction.

None of this is measured work. The base image is resolved here rather than lazily because a
container starts on its first exec, so an image the runtime still has to fetch would charge that
download to a `sandbox:start` span -- W times over in a fan-out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset" / "selected_topics.jsonl"
DOWNLOAD_CACHE = HERE / ".cache" / "beir"
DEFAULT_CORPUS_DIR = HERE / ".cache" / "corpus" / "trec-covid.test"

# Pinned by revision, not by tag, which upstream can repoint: an unpinned fetch would change the
# documents sitting under an already-committed selection. Corpus and qrels are separate repos.
CORPUS_DATASET = "BeIR/trec-covid"
CORPUS_REVISION = "7e16fde3016c639c7f856e803f4bab92645562c4"
_CORPUS_BASE = f"https://huggingface.co/datasets/{CORPUS_DATASET}/resolve/{CORPUS_REVISION}"
CORPUS_URL = f"{_CORPUS_BASE}/corpus/corpus-00000-of-00001.parquet"
CORPUS_SHA256 = "d76cea1b2304dbe67a1a54f7376a61de294976682a1d7d58d82de27141f3ba4a"
QUERIES_URL = f"{_CORPUS_BASE}/queries/queries-00000-of-00001.parquet"
QUERIES_SHA256 = "80bd564b1218a519ef0a396fa7b874941b7188d8240933e8d6fa867d7db59d6f"
QRELS_DATASET = "BeIR/trec-covid-qrels"
QRELS_REVISION = "532ac68ee6756ac22c9346eebf65bd3c6a042e10"
QRELS_SPLIT = "test"
QRELS_URL = f"https://huggingface.co/datasets/{QRELS_DATASET}/resolve/{QRELS_REVISION}/{QRELS_SPLIT}.tsv"
QRELS_SHA256 = "10669ab7d526cb04f52079139fd88c3d467a0776441b046567f540582798982b"

# Every cached name carries its dataset, since `test.tsv` alone would collide with another's.
CACHE_NAMES = {
    CORPUS_URL: "trec-covid-corpus-00000-of-00001.parquet",
    QUERIES_URL: "trec-covid-queries-00000-of-00001.parquet",
    QRELS_URL: "trec-covid-qrels-test.tsv",
}

# Kept graded, not collapsed to a boolean, because the draw's anchor is specifically a grade 2.
# -1 ("not judged" in some BEIR mirrors) is dropped by name; an unrecognised grade raises.
GRADES = (0, 1, 2)
EXCLUDED_GRADES = (-1,)

N_TOPICS = 8
# Also the maximum sweepable fan-out width: benchmark.py validates --widths against len(candidates).
CANDIDATES_PER_TOPIC = 8
# A judged docid can be absent from the shard, or carry a title with no abstract. Both are
# ineligible: a document summarisable in one turn shows up as fan-out skew that is not concurrency.
MIN_DOC_BYTES = 500
# Keeps a whole document inside one `read`: the read tool pages at 50 KB, so 32 K of text plus the
# newlines wrapping adds still arrives in a single call -- one tool:read span per item.
MAX_DOC_CHARS = 32_000
# Hard-wrapped on write, and not cosmetically: agency's `read` truncates any line over 2000
# characters without setting its `truncated` flag, and offset/limit are line-oriented, so no second
# call recovers the rest. A TREC-COVID abstract is usually one long line.
WRAP_COLUMNS = 100
SELECTION_SEED = 24
TITLE_SEPARATOR = "\n\n"

# Agency's own default image; nothing is added to it. The documents arrive as a read-only bind
# mount rather than an image layer -- which document an agent may read differs per agent.
DEFAULT_BASE_IMAGE = "agency-sandbox:latest"

# Resolved the way Agency's get_container_runtime() does -- podman first when both are usable --
# because the runtimes keep separate image stores, so asking the other records an unused image.
CONTAINER_RUNTIMES = ("podman", "docker")

MANIFEST_VERSION = 1
ROW_BATCH = 8192


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command and capture its output, for checks that parse it.

    A missing binary comes back non-zero rather than as a raise: "no container runtime is
    installed" is what preflight exists to report, and a FAIL line says it better than a traceback.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(cmd, 127, "", f"{type(e).__name__}: {e}")


def container_runtime() -> "str | None":
    """The runtime CLI Agency's sandbox backend will use, or None if there is none.

    Same order and reachability probe as Agency's get_container_runtime(): installed is not enough,
    since a runtime whose daemon is down is one Agency skips.
    """
    for runtime in CONTAINER_RUNTIMES:
        if shutil.which(runtime) and _run([runtime, "info"], timeout=120).returncode == 0:
            return runtime
    return None


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def _report(failures: int) -> int:
    if failures:
        print(f"{failures} check(s) failed.")
    else:
        print("All checks passed.")
    return 1 if failures else 0


def _longest_line(text: str) -> str:
    """The longest line of *text*, as a substring worth testing a read against."""
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    return max(lines, key=len) if lines else ""


def load_topics() -> list[dict]:
    """Read the pinned topic selection. benchmark.py keeps a copy, so a run never imports this."""
    if not DATASET_FILE.is_file():
        sys.exit(f"No pinned dataset at {DATASET_FILE}. Run: python prepare.py --dataset")
    with DATASET_FILE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def download(url: str, sha256: str) -> Path:
    """Fetch *url* into the cache once, renamed from `.part` only once its digest matches, so an
    interrupted transfer cannot leave a short file that the next run reads as a complete one."""
    dest = DOWNLOAD_CACHE / CACHE_NAMES[url]
    if dest.is_file():
        return dest

    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    print(f"Fetching {url.rsplit('/', 1)[-1]} ...")
    with urllib.request.urlopen(url, timeout=300) as resp, part.open("wb") as fh:  # noqa: S310 - pinned https URL
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            fh.write(chunk)
            size += len(chunk)

    got = digest.hexdigest()
    if got != sha256:
        part.unlink(missing_ok=True)
        sys.exit(f"checksum mismatch for {url}\n  expected {sha256}\n  got      {got}")
    part.replace(dest)
    print(f"  cached at {dest} ({size / 1e6:.1f} MB)")
    return dest


def parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Graded judgments per topic, from a BEIR `test.tsv`: `{qid: {docid: grade}}`."""
    judged: dict[str, dict[str, int]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t") if lines else []
    expected = ["query-id", "corpus-id", "score"]
    if [c.strip() for c in header] != expected:
        sys.exit(
            f"unexpected qrels header in {path}\n  expected {expected}\n  got      {header}\n"
            "The BEIR qrels schema has changed; update parse_qrels in prepare.py."
        )
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            sys.exit(f"{path}:{lineno}: expected 3 tab-separated fields, got {len(fields)}")
        qid, docid, score = (f.strip() for f in fields)
        grade = int(score)
        if grade in EXCLUDED_GRADES:
            continue
        if grade not in GRADES:
            sys.exit(
                f"{path}:{lineno}: grade {grade} is neither in GRADES={GRADES} nor in "
                f"EXCLUDED_GRADES={EXCLUDED_GRADES}. Decide which it is in prepare.py rather "
                "than letting it be scored as irrelevant by accident."
            )
        judged.setdefault(qid, {})[docid] = grade
    return judged


def read_queries(path: Path) -> dict[str, str]:
    """Topic text per query id, from a BEIR queries parquet."""
    import pyarrow.parquet as pq

    queries: dict[str, str] = {}
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=ROW_BATCH, columns=["_id", "text"]):
        # By name, not position: iter_batches does not promise the column order requested, and
        # swapping _id with text would produce a silently empty selection.
        ids = batch.column(batch.schema.get_field_index("_id")).to_pylist()
        texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
        for qid, text in zip(ids, texts):
            queries[qid] = (text or "").strip()
    return queries


def join_document(title: "str | None", text: "str | None") -> str:
    """One document as an agent reads it. Capped first, wrapped second: the cap belongs to the
    content, and wrapping only adds the newlines `read` needs (see WRAP_COLUMNS)."""
    body = f"{title or ''}{TITLE_SEPARATOR}{text or ''}"[:MAX_DOC_CHARS]
    return "\n".join(
        textwrap.fill(para, WRAP_COLUMNS) if para.strip() else ""
        for para in body.split("\n")
    )


def scan_corpus(path: Path, wanted: "set[str]", with_text: bool) -> dict:
    """One pass over the corpus shard, keeping only the docids in *wanted*.

    Streamed by row batch: the shard is 110 MB of 171k abstracts and at most 64 are needed. With
    *with_text* false only each document's byte length is kept, which is what decides eligibility.
    """
    import pyarrow.parquet as pq

    found: dict[str, object] = {}
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=ROW_BATCH, columns=["_id", "title", "text"]):
        ids = batch.column(batch.schema.get_field_index("_id")).to_pylist()
        # Gate on the cheap id column: converting a batch of abstracts costs far more than the
        # ids that decide whether any of them is wanted.
        if not any(docid in wanted for docid in ids):
            continue
        titles = batch.column(batch.schema.get_field_index("title")).to_pylist()
        texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
        for docid, title, text in zip(ids, titles, texts):
            if docid not in wanted or docid in found:
                continue
            doc = join_document(title, text)
            found[docid] = doc if with_text else len(doc.encode())
    return found


def corpus_digest(topics: list[dict], corpus_dir: Path) -> str:
    """The corpus's identity: sha256 over the sorted (qid, label, docid, file sha256) quadruples.

    Reads the written files rather than the parquet, so a document edited or truncated on disk
    after extraction changes it. benchmark.py re-verifies it before starting a container.
    """
    digest = hashlib.sha256()
    for topic in sorted(topics, key=lambda t: t["qid"]):
        for cand in topic["candidates"]:
            path = document_path(corpus_dir, topic["qid"], cand["label"])
            body = path.read_bytes() if path.is_file() else b""
            digest.update(
                f"{topic['qid']}\0{cand['label']}\0{cand['docid']}\0"
                f"{hashlib.sha256(body).hexdigest()}\0".encode()
            )
    return digest.hexdigest()


def document_dir(corpus_dir: Path, qid: str, label: str) -> Path:
    """The host directory mounted into one summariser's sandbox -- one per document, since a bind
    mount names a directory, and that is what makes a summariser unable to reach another's."""
    return corpus_dir / qid / label


def document_path(corpus_dir: Path, qid: str, label: str) -> Path:
    return document_dir(corpus_dir, qid, label) / "document.txt"


def load_manifest(corpus_dir: Path) -> dict:
    path = corpus_dir / "manifest.json"
    if not path.is_file():
        sys.exit(f"No corpus manifest at {path}. Run: python prepare.py --corpus")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Validate the five assumptions the whole benchmark rests on."""
    print("Preflight checks\n")
    failures = 0
    corpus_dir = args.corpus_dir

    # 1. The pin is readable and anchored. candidates[:W] IS the wave at width W, so candidates[0]
    #    is the item measured at every width and every ratio divides by its W=1 latency.
    print(f"[1/5] pinned selection ({DATASET_FILE.name})")
    topics: list[dict] = []
    try:
        topics = load_topics()
    except (json.JSONDecodeError, SystemExit) as e:
        _fail(f"could not read the pin: {e}")
        return 1
    if not topics:
        _fail("the pin is empty -- run `python prepare.py --dataset --force`")
        return 1
    unanchored = [t["qid"] for t in topics if not t["candidates"] or t["candidates"][0]["grade"] != 2]
    thin = [t["qid"] for t in topics if len(t["candidates"]) < CANDIDATES_PER_TOPIC]
    if unanchored:
        _fail(f"topic(s) {', '.join(unanchored)} have no grade-2 document first; the W=1 "
              "baseline would run on an unjudged item. Re-pin: python prepare.py --dataset --force")
        failures += 1
    elif thin:
        _fail(f"topic(s) {', '.join(thin)} carry fewer than {CANDIDATES_PER_TOPIC} candidates, "
              "so the widest sweep cannot run")
        failures += 1
    else:
        _ok(f"{len(topics)} topics x {CANDIDATES_PER_TOPIC} candidates, every wave anchored "
            "on a grade-2 document")

    # 2. Every pinned document is on disk with the bytes the manifest recorded.
    print("\n[2/5] corpus materialized and unchanged")
    manifest: dict = {}
    try:
        manifest = load_manifest(corpus_dir)
    except SystemExit as e:
        _fail(str(e))
        return 1
    missing = [
        f"{t['qid']}/{c['label']}"
        for t in topics
        for c in t["candidates"]
        if not document_path(corpus_dir, t["qid"], c["label"]).is_file()
    ]
    if missing:
        _fail(f"{len(missing)} document(s) missing, e.g. {', '.join(missing[:3])} -- "
              "run `python prepare.py --corpus --force`")
        failures += 1
    else:
        digest = corpus_digest(topics, corpus_dir)
        if digest != manifest.get("digest"):
            _fail(f"corpus digest {digest[:12]} does not match the manifest's "
                  f"{str(manifest.get('digest'))[:12]} -- the documents changed on disk")
            failures += 1
        else:
            _ok(f"{manifest['documents']} documents, {manifest['bytes'] / 1e3:.0f} KB, "
                f"digest {digest[:12]}")

    # 3. The base image an agSandbox will be built from is present locally.
    print("\n[3/5] sandbox base image")
    image = manifest.get("base_image") or args.base_image
    runtime = container_runtime()
    if runtime is None:
        _fail(f"no container runtime reachable ({' or '.join(CONTAINER_RUNTIMES)}); Agency's "
              "sandbox backend has nothing to run an agSandbox on")
        failures += 1
        print()
        return _report(failures)
    if _run([runtime, "image", "inspect", image]).returncode == 0:
        _ok(f"{image} present to {runtime} ({str(manifest.get('base_image_id'))[:19]})")
    else:
        _fail(f"{image} not present to {runtime}. It is built from the Agency source tree, "
              "not pulled:\n         cd /path/to/agency-staging && ./images/build.sh")
        failures += 1
        print()
        return _report(failures)

    # 4. The check that matters most: an empty or misplaced mount does not raise, `read` just
    #    answers "Not found", and every summary in the run would then be invented while every
    #    timing number still looked healthy.
    print("\n[4/5] read-only corpus mount, inside a live sandbox")
    if missing:
        _fail("skipped -- documents are missing (see above)")
        failures += 1
    else:
        try:
            from agency import agSandbox, agdata, agprof
            from agency.agconfig import agConfig
            from agency.agsandbox import agSandboxConfig
            from agency.tools import make_read

            import benchmark

            probe = topics[0]
            label = probe["candidates"][0]["label"]
            host_dir = document_dir(corpus_dir, probe["qid"], label)
            expected = document_path(corpus_dir, probe["qid"], label).read_text(encoding="utf-8")

            with tempfile.TemporaryDirectory() as td:
                cfg = agConfig(agSandboxConfig(base_image=image))
                agSandboxConfig(cfg).add_mount(
                    "corpus", host_dir, benchmark.CORPUS_MOUNT, "ro"
                )
                sandbox = agSandbox("preflight", agconfig=cfg)
                try:
                    with agprof.session(Path(td) / "prof"):
                        listing, _ = sandbox.exec(f"ls -1 {benchmark.CORPUS_MOUNT}")
                        read_tool = make_read(sandbox)
                        result = read_tool(agdata(file_path=benchmark.DOC_PATH)).to_dict()
                    metrics = agprof.summary_metrics() or {}

                    # Exactly one file: a mount of the whole corpus tree would also pass a "can it
                    # read its document" test while leaving every other document reachable.
                    entries = sorted(listing.split())
                    if entries == ["document.txt"]:
                        _ok(f"{benchmark.CORPUS_MOUNT} holds exactly document.txt")
                    else:
                        _fail(f"{benchmark.CORPUS_MOUNT} holds {entries} -- expected only "
                              "document.txt; each agent must reach exactly one document")
                        failures += 1

                    body = str(result.get("content") or result.get("error") or "")
                    # Compared on a sentence of the document, not on length: `read` returns
                    # line-numbered content, so byte counts differ while the prose must not.
                    probe_text = _longest_line(expected)
                    if probe_text and probe_text in body:
                        _ok(f"read returned the document's own text ({len(body)} chars)")
                    else:
                        _fail(f"read did not return the mounted document: {body[:200]!r}")
                        failures += 1

                    out, _ = sandbox.exec(f"touch {benchmark.CORPUS_MOUNT}/canary 2>&1 || true")
                    if document_dir(corpus_dir, probe["qid"], label).joinpath("canary").exists():
                        _fail("the mount is writable -- an agent could edit the document it "
                              "is being scored against; mode must be 'ro'")
                        failures += 1
                    else:
                        _ok(f"mount is read-only ({out.strip()[:60] or 'write refused'})")

                    rows = metrics.get("sandbox_metrics") or []
                    if rows:
                        _ok(f"agprof sampled the container cgroup ({len(rows)} sandbox row(s))")
                    else:
                        _fail("agprof produced no sandbox_metrics -- per-container CPU and I/O "
                              "will read null in every result")
                        failures += 1
                finally:
                    sandbox.destroy()
        except Exception as e:
            _fail(f"{type(e).__name__}: {e}")
            print("       If this is a cgroup/PID error, you are probably running the driver")
            print("       inside a container. Run natively on the host -- see README.md.")
            failures += 1

    # 5. The selected model is actually reachable.
    print("\n[5/5] model reachable")
    try:
        import benchmark

        # Unset options fall back to benchmark.py's, so preflight probes what the run will use.
        backend = args.backend or benchmark.DEFAULT_BACKEND
        model = args.model or benchmark.DEFAULT_MODEL
        region = args.region or benchmark.DEFAULT_REGION
        context_limit = args.context_limit or benchmark.DEFAULT_CONTEXT_LIMIT
        print(f"       {backend} / {model} / {args.base_url or 'no base_url'}")
        cfg = benchmark.build_llm_config(backend, model, region, context_limit,
                                        args.base_url, args.reasoning_effort)
        reply = benchmark.probe_model(cfg, base_image=image)
        _ok(f"model responded ({reply!r})")
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        failures += 1

    print()
    return _report(failures)


# --------------------------------------------------------------------------
# --dataset
# --------------------------------------------------------------------------
def cmd_dataset(args: argparse.Namespace) -> int:
    """Select N topics and pin each one's candidate documents."""
    if DATASET_FILE.is_file() and not args.force:
        print(f"{DATASET_FILE} already exists -- nothing to do. Use --force to reselect.")
        return 0

    qrels_file = download(QRELS_URL, QRELS_SHA256)
    queries_file = download(QUERIES_URL, QUERIES_SHA256)
    corpus_file = download(CORPUS_URL, CORPUS_SHA256)

    judged = parse_qrels(qrels_file)
    queries = read_queries(queries_file)
    print(f"\n{len(judged)} judged topics, {len(queries)} queries")

    # Eligibility resolved against the corpus before selecting, not after: selecting first and
    # discovering an absent docid during --corpus would mean a pin that cannot be materialized,
    # and the whole point of the pin is that it always can.
    pool_ids = {docid for qid in judged if qid in queries for docid in judged[qid]}
    print(f"Scanning {corpus_file.name} for {len(pool_ids)} judged documents ...")
    sizes = scan_corpus(corpus_file, pool_ids, with_text=False)
    print(f"  {len(sizes)} resident, "
          f"{sum(1 for n in sizes.values() if n >= MIN_DOC_BYTES)} at least {MIN_DOC_BYTES} B")

    eligible: dict[str, list[dict]] = {}
    for qid, grades in judged.items():
        if not queries.get(qid):
            continue
        rows = [
            {"docid": docid, "grade": grade}
            for docid, grade in grades.items()
            if sizes.get(docid, 0) >= MIN_DOC_BYTES
        ]
        if len(rows) >= args.candidates and any(r["grade"] == 2 for r in rows):
            eligible[qid] = rows

    # Ordered as strings, because a BEIR query id is a string: `--topics 8` selects qids
    # 1, 10, 11, 12, ... and not 1..8, the alternative being an ordering the dataset does not have.
    chosen = sorted(eligible)[: args.topics]
    if len(chosen) < args.topics:
        sys.exit(
            f"only {len(chosen)} topics have {args.candidates} eligible documents including a "
            f"grade-2 one; asked for {args.topics}. Lower --topics or --candidates."
        )

    print(f"\nSelecting {len(chosen)} topics (seed={args.seed}, "
          f"{args.candidates} candidates each):")
    selected: list[dict] = []
    for qid in chosen:
        rows = eligible[qid]
        # Sorted by docid before shuffling, so the draw depends only on the seed and not on qrels
        # row order. Seeded per topic, so changing --topics does not re-draw earlier topics.
        rng = random.Random(f"{args.seed}:{qid}")
        pool = sorted(rows, key=lambda r: r["docid"])
        rng.shuffle(pool)

        # A grade-2 document anchored at position 0, then a prefix. PREFIX-STABLE: item 0 is the
        # same document at every width, which is what makes throughput_gain compare the same work.
        # ANCHORED: the W=1 baseline it divides by is never a title-only stub.
        anchor = next(r for r in pool if r["grade"] == 2)
        draw = [anchor] + [r for r in pool if r is not anchor]
        draw = draw[: args.candidates]

        candidates = [
            {"label": f"doc_{i:02d}", "docid": r["docid"], "grade": r["grade"]}
            for i, r in enumerate(draw)
        ]
        relevant = sum(1 for c in candidates if c["grade"] > 0)
        print(f"  qid {qid!r}: {len(rows)} eligible of {len(judged[qid])} judged, "
              f"{relevant}/{len(candidates)} candidates relevant")
        selected.append(
            {
                "qid": qid,
                "query": queries[qid],
                "candidates": candidates,
                "pool_size": len(rows),
                "pool_relevant": sum(1 for r in rows if r["grade"] > 0),
                "candidate_relevant": relevant,
                "selection_seed": args.seed,
                "corpus_dataset": CORPUS_DATASET,
                "corpus_revision": CORPUS_REVISION,
                "qrels_dataset": QRELS_DATASET,
                "qrels_revision": QRELS_REVISION,
            }
        )

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": this file is committed, and a Windows run would otherwise rewrite it as CRLF.
    with DATASET_FILE.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in selected:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(selected)} topics to {DATASET_FILE}")
    print("Commit this file -- it is the pinned source of truth for the benchmark.")
    print("It carries docids and grades only, never document text: the text is fetched by "
          "--corpus and\nstays out of git.")
    return 0


# --------------------------------------------------------------------------
# --corpus
# --------------------------------------------------------------------------
def cmd_corpus(args: argparse.Namespace) -> int:
    """Materialize one text file per pinned candidate, and a manifest over them."""
    topics = load_topics()
    corpus_dir = args.corpus_dir
    wanted = {c["docid"] for t in topics for c in t["candidates"]}

    if (corpus_dir / "manifest.json").is_file() and not args.force:
        # Re-verify rather than trust: corpus_digest reads every document's bytes, so a file
        # edited since extraction is caught here rather than by an agent mid-episode.
        existing = json.loads((corpus_dir / "manifest.json").read_text())
        if existing.get("digest") == corpus_digest(topics, corpus_dir):
            print(f"{corpus_dir} already holds {existing['documents']} verified documents "
                  f"(digest {existing['digest'][:12]}) -- nothing to do.")
            return 0
        print(f"{corpus_dir} does not match the pin; re-extracting.")

    corpus_file = download(CORPUS_URL, CORPUS_SHA256)
    print(f"\nExtracting {len(wanted)} documents from {corpus_file.name} ...")
    docs = scan_corpus(corpus_file, wanted, with_text=True)
    if len(docs) < len(wanted):
        # Unreachable unless the pin and the pinned revision have drifted: --dataset only ever
        # pins docids it found resident in this shard.
        sys.exit(
            f"{len(wanted) - len(docs)} pinned docid(s) are not in {CORPUS_DATASET} at "
            f"{CORPUS_REVISION[:12]}: {sorted(wanted - set(docs))[:5]}\n"
            "The pin and the corpus revision have drifted. Re-pin: "
            "python prepare.py --dataset --force"
        )

    entries: dict[str, dict] = {}
    total_bytes = 0
    for topic in topics:
        per_topic: dict[str, dict] = {}
        for cand in topic["candidates"]:
            body = str(docs[cand["docid"]])
            path = document_path(corpus_dir, topic["qid"], cand["label"])
            path.parent.mkdir(parents=True, exist_ok=True)
            # newline="\n": the digest is over bytes, so CRLF would produce a corpus no Linux
            # run can reproduce.
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
            raw = path.read_bytes()
            total_bytes += len(raw)
            per_topic[cand["label"]] = {
                "docid": cand["docid"],
                "grade": cand["grade"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        entries[topic["qid"]] = per_topic
        sizes = [v["bytes"] for v in per_topic.values()]
        print(f"  qid {topic['qid']!r}: {len(sizes)} documents, "
              f"{min(sizes)}-{max(sizes)} B")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_dataset": CORPUS_DATASET,
        "corpus_revision": CORPUS_REVISION,
        "corpus_sha256": CORPUS_SHA256,
        "qrels_dataset": QRELS_DATASET,
        "qrels_revision": QRELS_REVISION,
        "min_doc_bytes": MIN_DOC_BYTES,
        "max_doc_chars": MAX_DOC_CHARS,
        "base_image": None,
        "base_image_id": None,
        "topics": len(topics),
        "documents": sum(len(v) for v in entries.values()),
        "bytes": total_bytes,
        "digest": None,
        "docs": entries,
    }
    manifest["digest"] = corpus_digest(topics, corpus_dir)
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {manifest['documents']} documents ({total_bytes / 1e3:.0f} KB) "
          f"to {corpus_dir}")
    print(f"Corpus digest: {manifest['digest']}")
    print("Not committed: .cache/ is gitignored, and the pin plus a pinned revision is "
          "enough to\nreproduce these bytes anywhere.")
    return 0


# --------------------------------------------------------------------------
# --image
# --------------------------------------------------------------------------
def cmd_image(args: argparse.Namespace) -> int:
    """Record which sandbox image the run will use, and its resolved image id.

    Does not build one -- agency-sandbox:latest is never published. Recording the id is what makes
    a result comparable across machines: the tag is mutable, the id is not.
    """
    manifest_path = args.corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"No corpus manifest at {manifest_path}. Run: python prepare.py --corpus")

    runtime = container_runtime()
    if runtime is None:
        print(f"No container runtime reachable ({' or '.join(CONTAINER_RUNTIMES)}). Agency's "
              "sandbox backend needs one; install it, then re-run this step.")
        return 1

    image = args.base_image
    inspected = _run([runtime, "image", "inspect", "--format", "{{.Id}}", image], timeout=120)
    if inspected.returncode != 0:
        print(f"{image} is not present to {runtime}. Build it from the Agency checkout:")
        print("    cd /path/to/agency-staging && ./images/build.sh")
        return 1

    image_id = inspected.stdout.strip()
    manifest = json.loads(manifest_path.read_text())
    manifest["base_image"] = image
    manifest["base_image_id"] = image_id
    manifest["container_runtime"] = runtime
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{image}\n  id: {image_id}\n  runtime: {runtime}\n  recorded in {manifest_path}")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="validate load-bearing assumptions")
    p.add_argument("--dataset", action="store_true", help="select and pin the topics and candidates")
    p.add_argument("--corpus", action="store_true", help="materialize the pinned documents")
    p.add_argument("--image", action="store_true", help="record the sandbox base image and its id")
    p.add_argument("--force", action="store_true",
                   help="with --dataset or --corpus, redo the work instead of skipping it")
    p.add_argument("--topics", type=int, default=N_TOPICS,
                   help=f"how many topics to pin (default {N_TOPICS}); requires --dataset --force")
    p.add_argument("--candidates", type=int, default=CANDIDATES_PER_TOPIC,
                   help=f"candidates per topic, and so the maximum sweepable fan-out width "
                        f"(default {CANDIDATES_PER_TOPIC}); requires --dataset --force")
    p.add_argument("--seed", type=int, default=SELECTION_SEED,
                   help=f"RNG seed for the candidate draw (default {SELECTION_SEED})")
    p.add_argument("--base-image", default=os.environ.get("BENCH_BASE_IMAGE", DEFAULT_BASE_IMAGE),
                   help=f"sandbox base image (default {DEFAULT_BASE_IMAGE})")
    p.add_argument("--corpus-dir", type=Path,
                   default=Path(os.environ.get("BENCH_CORPUS", DEFAULT_CORPUS_DIR)),
                   help="where the extracted documents live")
    # Left as None and resolved from benchmark.py inside --check. Importing benchmark here would
    # make agency a hard dependency of --dataset/--corpus/--image, none of which need it.
    p.add_argument("--backend", default=os.environ.get("BENCH_BACKEND"),
                   help="backend to probe in --check (default: benchmark.py's)")
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""),
                   help="the endpoint to probe in --check ($LLM_BASE_URL)")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"),
                   help="model to probe in --check ($LLM_MODEL, default: benchmark.py's)")
    p.add_argument("--reasoning-effort", default=os.environ.get("LLM_REASONING_EFFORT", ""),
                   help="reasoning effort to probe with ($LLM_REASONING_EFFORT)")
    p.add_argument("--region", default=os.environ.get("BENCH_REGION"))
    p.add_argument("--context-limit", type=int, default=None)
    args = p.parse_args()

    if args.candidates < 1:
        sys.exit(f"--candidates must be >= 1, got {args.candidates}")
    if args.topics < 1:
        sys.exit(f"--topics must be >= 1, got {args.topics}")

    # No flags: run everything, in dependency order.
    if not (args.check or args.dataset or args.corpus or args.image):
        args.check = args.dataset = args.corpus = args.image = True

    rc = 0
    # Dependency order: the pin decides what --corpus extracts, --image records itself into the
    # manifest --corpus wrote, and --check validates the finished result.
    if args.dataset:
        rc |= cmd_dataset(args)
        print()
    if args.corpus:
        rc |= cmd_corpus(args)
        print()
    if args.image:
        rc |= cmd_image(args)
        print()
    if args.check:
        check_rc = cmd_check(args)
        if check_rc:
            print("\nPreflight failed -- fix the above before continuing.")
        rc |= check_rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
