"""
Paper summarization benchmark -- targets concurrency.

A TREC-COVID topic is summarized by a wave of W forked agents, one per candidate paper, each
reading its own document from a read-only /corpus mount with a real tool call in its own
agSandbox. The pending summaries are then reduced by a single reporter agent. Because W=1 runs
through the same code path as W=8, what this measures is whether width buys overlap or only buys
tokens.

Every sandbox is created and started during warm-up, outside the measured window, so no image
resolution and no container start is inside any number reported here.

The primary artifact is the profiler trace under output/. The record beside it is narrow on
purpose: tokens, TTFT, per-run spans and per-sandbox cgroup rows are already in
profiler/summary.json, and every summary and quote in summaries.json.

Run:
    python prepare.py                          # once: pin, fetch corpus, record image, check
    python benchmark.py                        # one topic, widths 1,2,4,8
    python benchmark.py --topics 1 --widths 1  # the cheapest possible episode
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from agency import agSandbox, agdata, agent, agprof, agskill, agsync
from agency.agconfig import agConfig
from agency.agllm_backends import (
    agAnthropicBackendConfig,
    agBedrockBackendConfig,
    agOpenAIBackendConfig,
    agVLLMBackendConfig,
)
from agency.agsandbox import agSandboxConfig
from agency.agskill import agSkillConfig
from agency.tools import make_read

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset" / "selected_topics.jsonl"
DEFAULT_CORPUS_DIR = HERE / ".cache" / "corpus" / "trec-covid.test"
OUTPUT_DIR = HERE / "output"

# /corpus and deliberately not /workspace: agency's container backend uses /workspace as its
# working directory and re-creates it after commit/restore, so mounting over it collides.
CORPUS_MOUNT = "/corpus"
DOC_PATH = f"{CORPUS_MOUNT}/document.txt"

DEFAULT_BACKEND = "bedrock"
DEFAULT_MODEL = "minimax.minimax-m2.5"
DEFAULT_REGION = "us-east-2"
# Pinned, and not only to bound context: an unset context_limit makes agent.fork resolve it via
# backend.list_models(), putting a live HTTP round trip in front of every fork during warm-up.
DEFAULT_CONTEXT_LIMIT = 196000
# One read plus one submit is the whole job, so this is generous -- and tight, because a runaway
# agent is charged W times over in a single episode.
DEFAULT_MAX_STEPS = 24
# A sweep, not a single width: W=1 through the same code path is the only honest baseline, and the
# ratio between the ends of the sweep is the result. main() warns when 1 is missing.
DEFAULT_WIDTHS = "1,2,4,8"
DEFAULT_TOPICS = 1
# `dependency` dereferences each pending agdata in item order, which is what the paper-crawler app
# does; `barrier` calls agsync() and opens a visible agsync:join span. Both are real joins and
# should report the same makespan -- what differs is the shape in the trace.
JOIN_MODES = ("dependency", "barrier")
# A shorter quote grounds by accident: "COVID-19" appears in every document. Shorter quotes abstain.
MIN_QUOTE_CHARS = 24

# Agent harnesses this benchmark can drive. "native" is Agency's own in-process ReAct loop. The
# name is recorded in every result and forms the top level of the output path, so runs under
# different harnesses land side by side.
HARNESSES: "dict[str, object]" = {
    "native": lambda: [],
}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_api_key() -> str:
    """Read the single-line raw token from the benchmark suite root."""
    candidates = [
        Path(p)
        for p in (os.environ.get("AGENCY_BENCH_API_KEY_FILE"), "/run/secrets/api-key")
        if p
    ]
    candidates.append(HERE.parents[1] / "api-key")
    for path in candidates:
        if path.is_file():
            return path.read_text().strip()
    raise FileNotFoundError(f"api-key not found in any of: {[str(p) for p in candidates]}")


def build_llm_config(backend: str, model: str, region: str, context_limit: int):
    """Return the Agency backend config view for *backend*."""
    if backend == "bedrock":
        # The OpenAI-compatible Bedrock path reads this env var when no api_key kwarg
        # is passed. minimax.* is not an Anthropic model id, so it routes that way.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = load_api_key()
        return agBedrockBackendConfig(region=region, model=model, context_limit=context_limit)
    if backend == "openai":
        return agOpenAIBackendConfig(model=model, api_key=load_api_key(), context_limit=context_limit)
    if backend == "anthropic":
        return agAnthropicBackendConfig(model=model, api_key=load_api_key(), context_limit=context_limit)
    if backend == "vllm":
        return agVLLMBackendConfig(
            base_url=os.environ["LLM_BASE_URL"],
            model=model,
            api_key=os.environ.get("LLM_API_KEY", ""),
            context_limit=context_limit,
        )
    raise ValueError(f"unknown backend {backend!r}; valid: bedrock, openai, anthropic, vllm")


def probe_model(llm_config, base_image: "str | None" = None) -> str:
    """One trivial call, to confirm the model is reachable. Used by prepare.py --check.

    agskill provisions a sandbox even for a tool-less skill, so *base_image* reuses one known to
    exist locally; otherwise this fails on a missing image and says nothing about the model.
    """
    skill = agskill(
        name="probe",
        system_prompt="Reply with the single word: ok",
        replace_tools=[],
        input_schema=agdata(ping=str),
        output_schema=agdata(reply=str),
    )
    sources = [llm_config, agSkillConfig(react_max_steps=4)]
    if base_image:
        sources.append(agSandboxConfig(base_image=base_image))

    if agent.log_dir is None:
        agent.log_dir = Path(tempfile.mkdtemp(prefix="agency-bench-probe-"))

    ag = agent(agconfig=agConfig(*sources))
    try:
        data = ag.run(skill, agdata(ping="ping")).wait().to_dict()
    finally:
        if ag.sandbox is not None:
            try:
                ag.sandbox.destroy()
            except Exception:  # noqa: S110 -- teardown must not mask the probe result
                pass
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("reply", "")


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------
def build_summariser_skill(sandbox: agSandbox) -> agskill:
    """One document, read from the mount and summarized. Bound to this item's sandbox.

    replace_tools is fixed at construction and make_read needs a live sandbox, so this is rebuilt
    per item against a sandbox already assigned to that item's agent.
    """
    return agskill(
        name="paper_summariser",
        system_prompt=(
            "You are a biomedical research assistant summarizing ONE paper.\n\n"
            f"Read {DOC_PATH}. That file is the entire paper available to you: its first line "
            "is the title and the rest is the abstract. You have exactly one tool, `read`, and "
            "exactly one file to read -- there is nothing else on the filesystem to explore.\n\n"
            "If the read comes back truncated, call `read` again with an offset until you have "
            "reached the end of the document. Do not summarize a partial document.\n\n"
            "Then report:\n"
            "  - summary: 2-4 sentences on what the paper found.\n"
            "  - key_findings: up to 5 short factual claims from the paper.\n"
            "  - evidence_quote: a VERBATIM span of at least 30 characters copied exactly from "
            "the document, supporting your summary. Copy it character for character; do not "
            "paraphrase, reword or repair it. If the document contains nothing quotable, "
            "return an empty string rather than inventing one.\n"
            "  - relevant: true if this paper would help answer the research topic you were "
            "given, false otherwise. Judge the paper you actually read; a plausible-sounding "
            "guess is worse than a false.\n"
        ),
        replace_tools=[make_read(sandbox)],
        input_schema=agdata(label=str, query=str, document_path=str),
        output_schema=agdata(
            summary=str, key_findings=list[str], evidence_quote=str, relevant=bool
        ),
    )


def build_reducer_skill() -> agskill:
    """The fan-in: the topic plus the W summaries -> a report and a used/unused label ledger.

    Tool-less, but not sandbox-free: agskill provisions a sandbox lazily when ag.sandbox is None,
    which would put a container start inside the measured window, so warm_fleet binds the reducer
    one like every other agent. The ledger stays in the workload and out of the metrics -- it is
    what makes the reduce depend on the whole wave, and it lands in summaries.json.
    """
    return agskill(
        name="report_compiler",
        system_prompt=(
            "You are compiling a short evidence report for a research topic from summaries "
            "that other assistants produced, each from a different paper.\n\n"
            "Write a report of at most 250 words answering the topic from those summaries "
            "only. Do not add outside knowledge, and do not treat a summary marked "
            "relevant=false as support.\n\n"
            "Then account for EVERY label you were given, exactly once, across two lists:\n"
            "  - used_labels: labels whose content you drew on.\n"
            "  - unused_labels: labels you set aside, including any that arrived as errors.\n"
            "Every input label must appear in exactly one of the two lists."
        ),
        replace_tools=[],
        input_schema=agdata(topic=str, summaries=list[dict]),
        output_schema=agdata(report=str, used_labels=list[str], unused_labels=list[str]),
    )


def warm_fleet(topic: dict, items: list[dict], llm_config, args) -> dict:
    """Fork W summarisers plus one reducer, each with its own sandbox, all before the clock.

    A fresh agent per item, and reusing one is not an optimization: two run() calls on the same
    agent serialize on that agent's shared agcontext history future, so a reused agent would
    measure a queue rather than a wave.

    Every sandbox is started here, one at a time, so no sandbox:start lands inside the measured
    window. Serial on purpose: starting W at once would measure the runtime's start contention.
    None of it is timed -- the cost is visible as the gap between wall_clock_s and wall_ms.
    """
    # One lead agent carrying the config, then W+1 forks of it. The lead is never run and never
    # given a sandbox: agent.fork copies the source's sandbox only when it has one, so a
    # sandbox-less lead is what lets each fork be given its own mount below.
    lead = agent(
        agconfig=agConfig(
            llm_config,
            agSandboxConfig(base_image=args.base_image),
            agSkillConfig(react_max_steps=args.max_steps),
            *HARNESSES[args.harness](),
        )
    )

    fleet: dict = {
        "lead": lead,
        "workers": [],
        "skills": [],
        "sandboxes": [],
        "agnames": [],
        "reducer": None,
        "reducer_skill": build_reducer_skill(),
        "warm_failures": [],
    }

    # Reducer first, so workers[i] lines up with items[i] and the fan-out's warm-up is contiguous.
    plan = [(None, "reducer")] + [(it, it["label"]) for it in items]
    for item, label in plan:
        ag = agent.fork(lead)

        if item is not None:
            # Mutates this fork's own agconfig: agent.fork clones it, and agSandboxConfig wraps
            # an existing agConfig in place. Mounts are resolved eagerly by agSandbox.__init__,
            # so this must happen before the sandbox.
            host_dir = args.corpus / topic["qid"] / item["label"]
            agSandboxConfig(ag.agconfig).add_mount(
                "corpus", str(host_dir), CORPUS_MOUNT, "ro"
            )

        sandbox = agSandbox(ag.agname, agconfig=ag.agconfig)
        ag.sandbox = sandbox  # bind BEFORE building the skill; make_read closes over it
        fleet["sandboxes"].append(sandbox)
        fleet["agnames"].append(ag.agname)

        try:
            sandbox.exec("true", timeout=180)
        except Exception as e:
            # Collected rather than raised: a fleet one sandbox short still measures a wave, and
            # the failure resurfaces as that item's own error when its agent reads.
            fleet["warm_failures"].append(f"{label}: {type(e).__name__}: {e}")

        if item is None:
            fleet["reducer"] = ag
        else:
            fleet["workers"].append(ag)
            fleet["skills"].append(build_summariser_skill(sandbox))

    return fleet


def join_wave(pending: list, agents: list, mode: str) -> list[dict]:
    """Join the wave, rows in item order, a failure becoming a row rather than a raise.

    `barrier` opens an agsync:join span, and whether it appears in the trace is itself the check
    that this knob is not decorative. `dependency` blocks on each pending agdata in item order,
    so a slow item 0 delays the *observation* of item 7's result but not its execution -- which
    is why per-item latency is read back from the agents' own logs and never from here.

    One item's failure must not cost the other W-1 items' measurements, already paid for.
    """
    if mode == "barrier":
        agsync(*agents)

    rows = []
    for p in pending:
        try:
            rows.append(p.wait().to_dict())
        except Exception as e:
            rows.append({"error": f"{type(e).__name__}: {e}"})
    return rows


# --------------------------------------------------------------------------
# measurement -- reading the wave back
# --------------------------------------------------------------------------
def read_item_runs(log_dir: Path, agnames: list[str]) -> list[dict]:
    """Per-item start and end, parsed out of each agent's own timeline jsonl.

    Read after the fact rather than timed in-process, and that is the whole design: ag.run()
    returns immediately for all W items, so host-side stamps could only ever recover the wave's
    maximum, never its shape. Both join modes read identically.
    """
    runs = []
    for agname in agnames:
        row: dict = {"agname": agname, "start": None, "end": None, "ms": None}
        path = log_dir / f"{agname}_timeline.jsonl"
        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line: the run was killed mid-write
                # First skill entry only: an agent runs exactly one skill here, so a second entry
                # would mean the fleet was reused -- a queue rather than a wave.
                if entry.get("type") == "skill" and row["start"] is None:
                    row["start"] = _epoch(entry.get("ts_start"))
                    row["end"] = _epoch(entry.get("ts_end"))
        except OSError:
            runs.append(row)
            continue

        if row["start"] is not None and row["end"] is not None:
            row["ms"] = round((row["end"] - row["start"]) * 1000, 2)
        runs.append(row)
    return runs


def _epoch(stamp: "str | None") -> "float | None":
    """An aglog ISO-8601 stamp as epoch seconds, for comparison against time.time() marks."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return None


def summarize_wave(runs: list[dict], fanout_ms: float) -> dict:
    """One wave's shape: item latency, peak overlap, and its cost over its slowest item."""
    clocked = [r for r in runs if r["ms"] is not None]
    out: dict = {"item_ms_p50": None, "peak_overlap": None, "makespan_ratio": None}
    if not clocked:
        return out
    # An item whose timeline could not be read is missing from the median and the overlap sweep
    # alike, so a lost log quietly *lowers* peak_overlap. Printed: a warning about the run.
    if len(clocked) != len(runs):
        print(f"   warning: {len(runs) - len(clocked)} of {len(runs)} items have no readable "
              "timeline; item_ms_p50 and peak_overlap cover only the rest")

    durations = sorted(r["ms"] for r in clocked)
    out["item_ms_p50"] = round(statistics.median(durations), 2)

    # Sweep line over start/end events. -1 sorts before +1 at an identical timestamp, so an item
    # ending exactly as another starts is a handoff, not an overlap -- ms stamps would say 2.
    events = sorted([(r["start"], 1) for r in clocked] + [(r["end"], -1) for r in clocked])
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    out["peak_overlap"] = peak

    # The critical path of a single-stage fan-out is its slowest item, so makespan_ratio == 1.0
    # means the wave cost exactly that -- the best a fan-out can do. durations[-1] is not recorded.
    out["makespan_ratio"] = round(fanout_ms / durations[-1], 3) if durations[-1] else None
    return out


# --------------------------------------------------------------------------
# scoring -- secondary signal only
# --------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Collapse whitespace and case.

    A quote differing from the document only in wrapping is a real quote a line-oriented read
    reflowed; scoring it as a fabrication would make grounding_rate measure the tool.
    """
    return " ".join((text or "").split()).lower()


def score_grounding(items: list[dict], rows: list[dict], corpus_dir: Path,
                    qid: str) -> "float | None":
    """The share of evidence quotes that are really in the document, or None if none was checkable.

    The one quality number kept: the only thing standing between "W agents overlapped" and "W
    agents overlapped while inventing text". Scored against the document's own bytes, read on the
    host, so no confident summary can talk its way past it. A quote under MIN_QUOTE_CHARS abstains
    rather than fails, which is why None (nothing was scorable) is not the same as 0.0.
    """
    grounded = checked = 0
    for item, row in zip(items, rows):
        if "error" in row:
            continue
        quote = row.get("evidence_quote") or ""
        if len(quote.strip()) < MIN_QUOTE_CHARS:
            continue
        checked += 1
        doc = corpus_dir / qid / item["label"] / "document.txt"
        try:
            body = doc.read_text(encoding="utf-8")
        except OSError:
            body = ""
        if body and _normalize(quote) in _normalize(body):
            grounded += 1
    return round(grounded / checked, 4) if checked else None


# --------------------------------------------------------------------------
# run loop
# --------------------------------------------------------------------------
def load_topics() -> list[dict]:
    """Pinned topics. Own loader, deliberately: running the benchmark must not import prepare."""
    if not DATASET_FILE.is_file():
        sys.exit(f"No pinned dataset at {DATASET_FILE}. Run: python prepare.py --dataset")
    with DATASET_FILE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_corpus(pinned: list[dict], corpus_dir: Path) -> dict:
    """Re-verify the corpus against its manifest, before the first sandbox.

    Takes every pinned topic, never the selected subset: the manifest's digest is over the whole
    corpus, so a subset could only ever mismatch. Checked here because warming a fleet costs tens
    of seconds and W+1 sandboxes, and a half-built corpus would surface only after paying for it.
    """
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"No corpus at {corpus_dir}. Run: python prepare.py --corpus")
    manifest = json.loads(manifest_path.read_text())

    digest = hashlib.sha256()
    for topic in sorted(pinned, key=lambda t: t["qid"]):
        for cand in topic["candidates"]:
            path = corpus_dir / topic["qid"] / cand["label"] / "document.txt"
            body = path.read_bytes() if path.is_file() else b""
            digest.update(
                f"{topic['qid']}\0{cand['label']}\0{cand['docid']}\0"
                f"{hashlib.sha256(body).hexdigest()}\0".encode()
            )
    got = digest.hexdigest()
    if got != manifest.get("digest"):
        sys.exit(
            f"corpus digest mismatch at {corpus_dir}\n"
            f"  manifest {manifest.get('digest')}\n  on disk  {got}\n"
            "Re-extract: python prepare.py --corpus --force"
        )

    # The filesystem the documents sit on: a 9p mount (a Windows path reached from WSL) pays a
    # bridge on every read, which would be charged to the wave and read as contention.
    fstype = None
    try:
        probe = subprocess.run(["stat", "-f", "-c", "%T", str(corpus_dir)],
                               capture_output=True, text=True, timeout=30)
        fstype = probe.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    if fstype in ("9p", "v9fs", "cifs", "smb2", "fuseblk", "nfs"):
        print(f"  WARNING: the corpus is on a {fstype} mount. Every tool:read pays a "
              "filesystem\n  bridge, which lands inside the measured window. Move the corpus "
              "to a native\n  filesystem (BENCH_CORPUS=...) before trusting any timing here.")

    return {"digest": got, "fstype": fstype,
            "base_image_id": manifest.get("base_image_id"),
            "manifest_base_image": manifest.get("base_image"),
            # Whichever runtime prepare.py resolved the image against: part of what produced
            # these numbers, not what they are about.
            "container_runtime": manifest.get("container_runtime")}


def run_episode(topic: dict, width: int, trial: int, llm_config, args, corpus: dict,
                ep_out: Path) -> dict:
    """One topic at one width under one profiler session. This function IS the timing boundary.

    Everything expensive that is not the thing being measured -- forking, container starts, skill
    construction, teardown -- happens outside the three marks. Inside: W non-blocking dispatches,
    the join, the reduce.
    """
    ep_out.mkdir(parents=True, exist_ok=True)
    items = topic["candidates"][:width]

    # Built before the try so a failed episode still produces a row in the aggregate.
    record = {
        "qid": topic["qid"],
        "width": width,
        "trial": trial,
        "harness": args.harness,
        "backend": args.backend,
        "model": args.model,
        "join_mode": args.join_mode,
        "max_steps": args.max_steps,
        "base_image": args.base_image,
        "base_image_id": corpus["base_image_id"],
        "container_runtime": corpus["container_runtime"],
        "corpus_digest": corpus["digest"],
        "corpus_fstype": corpus["fstype"],
        "agnames": [],
        "status": "not_started",
    }

    # Per-episode log dir on the agent classvar; agent.fork resolves log_dir per agent, so every
    # fork writes its own <agname>_timeline.jsonl here -- read_item_runs' only source.
    log_dir = ep_out / "agent_logs"
    agent.log_dir = log_dir

    # Three marks on one clock, two consecutive intervals whose sum is the third, so
    # wall_ms == fanout_ms + reduce_ms is an identity a reader can check rather than a residual.
    marks: dict = {}

    def mark(name: str) -> None:
        marks[name] = time.perf_counter()

    fleet = None
    started = time.perf_counter()
    try:
        fleet = warm_fleet(topic, items, llm_config, args)
        record["agnames"] = fleet["agnames"]
        # Printed, not recorded: it resurfaces in the record as that item's error anyway.
        for failure in fleet["warm_failures"]:
            print(f"   warning: sandbox did not warm: {failure}")

        with agprof.session(ep_out / "profiler"):
            # A sandbox registers its cgroup only while a session is active, and the tool-less
            # reducer never touches its container in-window. Touching every sandbox here, before
            # the first mark, is charged to no phase and makes len(sandbox_metrics) read W+1.
            for sandbox in fleet["sandboxes"]:
                try:
                    sandbox.exec("true", timeout=180)
                except Exception as e:
                    print(f"   warning: sandbox registration touch failed: {e}")
            # agprof.annotate attaches to the innermost ACTIVE span and no-ops without one, so
            # the episode's metadata needs a span to live on.
            with agprof.span(f"paper_summarization:episode[W{width}]"):
                agprof.annotate(qid=topic["qid"], width=width, trial=trial,
                                join_mode=args.join_mode, harness=args.harness,
                                model=args.model)

                mark("fanout_start")
                with agprof.span(f"paper_summarization:fanout[{width}]"):
                    pending = [
                        ag.run(sk, agdata(label=it["label"], query=topic["query"],
                                          document_path=DOC_PATH))
                        for ag, sk, it in zip(fleet["workers"], fleet["skills"], items)
                    ]
                    rows = join_wave(pending, fleet["workers"], args.join_mode)
                mark("fanout_end")

                with agprof.span("paper_summarization:reduce"):
                    reduced = fleet["reducer"].run(
                        fleet["reducer_skill"],
                        agdata(
                            topic=topic["query"],
                            summaries=[
                                {"label": it["label"],
                                 **{k: v for k, v in r.items() if k != "key_findings"},
                                 "key_findings": r.get("key_findings") or []}
                                for it, r in zip(items, rows)
                            ],
                        ),
                    ).wait().to_dict()
                mark("reduce_end")

        # ---- the three marks, two intervals, one identity ----
        fanout_ms = (marks["fanout_end"] - marks["fanout_start"]) * 1000
        reduce_ms = (marks["reduce_end"] - marks["fanout_end"]) * 1000
        record.update({
            "wall_ms": round((marks["reduce_end"] - marks["fanout_start"]) * 1000, 2),
            "fanout_ms": round(fanout_ms, 2),
            "reduce_ms": round(reduce_ms, 2),
        })

        runs = read_item_runs(log_dir, [a.agname for a in fleet["workers"]])
        record.update(summarize_wave(runs, fanout_ms))

        # No failure count: status already distinguishes none, some, all; rows go to summaries.json.
        errors = [r for r in rows if "error" in r]
        record["status"] = (
            "ok" if not errors
            else "max_steps_exceeded" if any("max_steps" in str(r["error"]) for r in errors)
            else "partial" if len(errors) < len(rows)
            else "error"
        )
        if errors:
            record["error"] = str(errors[0]["error"])

        record["grounding_rate"] = score_grounding(items, rows, args.corpus, topic["qid"])
        # Every summary, quote and the reducer's whole report, next to the trace that timed them.
        (ep_out / "summaries.json").write_text(json.dumps(
            {"items": items, "rows": rows, "report": reduced}, indent=2))
    except Exception as e:
        record["status"] = "exception"
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        record["wall_clock_s"] = round(time.perf_counter() - started, 2)
        if fleet is not None:
            # Explicit, and outside every mark and the session: Agency's live-agent registry is a
            # WeakSet, so an implicit release would be collected inside the next episode's clock.
            for sandbox in fleet["sandboxes"]:
                try:
                    sandbox.destroy()
                except Exception as e:  # teardown must not mask the episode's result
                    print(f"   warning: sandbox teardown failed: {e}")

    # Nothing is copied back out of profiler/summary.json. Two things there are worth reading by
    # hand when a result looks wrong: len(sandbox_metrics) should be W+1, and run_metrics.p50_ms is
    # NOT a cross-check on item_ms_p50 -- it medians all W+1 runs, the reducer included. The wave's
    # second source is the trace's run<N>:*:paper_summariser spans.
    (ep_out / "result.json").write_text(json.dumps(record, indent=2))
    return record


def write_aggregate(records: list[dict], out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(records, indent=2))

    def fmt(value, spec=".2f", scale=1.0):
        return "n/a" if value is None else format(value * scale, spec)

    lines = [
        "# Paper summarization results",
        "",
        "Profiler traces are the primary artifact; `Grounded` is a secondary sanity signal.",
        "`Peak ovl` is the claim: it should equal W. `Throughput gain` is the only speedup",
        "figure here and reads n/a unless the same run measured a status=ok W=1 row for the",
        "same topic; `Makespan` is not a speedup. `Wall` is `Fan-out` + `Reduce` by construction.",
        "",
        "| Topic | W | Trial | Status | Wall (s) | Fan-out (s) | Reduce (s) | Item p50 (s) "
        "| Peak ovl | Makespan | Grounded |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r['qid']}` | {r['width']} | {r['trial']} | {r['status']} | "
            f"{fmt(r.get('wall_ms'), '.1f', 1e-3)} | {fmt(r.get('fanout_ms'), '.1f', 1e-3)} | "
            f"{fmt(r.get('reduce_ms'), '.1f', 1e-3)} | "
            f"{fmt(r.get('item_ms_p50'), '.1f', 1e-3)} | "
            f"{r.get('peak_overlap') or 'n/a'} | {fmt(r.get('makespan_ratio'))} | "
            f"{fmt(r.get('grounding_rate'), '.0%')} |"
        )

    # Per topic, not per run: item 0 is the same document at every width only within one topic, so a
    # median across topics would compare document lengths and call it concurrency. status == "ok" is
    # load-bearing, not tidiness -- a failed item is still timed, and what it times is its own
    # retry backoff, large enough as a denominator to publish a fictional speedup.
    baseline: dict = {}
    for r in records:
        if r["width"] == 1 and r["status"] == "ok" and r.get("item_ms_p50"):
            baseline.setdefault(r["qid"], []).append(r["item_ms_p50"])

    by_width: dict = {}
    for r in records:
        by_width.setdefault(r["width"], []).append(r)

    lines += [
        "", "## By width", "",
        "| W | N | Item p50 (s) | Fan-out (s) | Peak ovl | Makespan | Throughput gain | Grounded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for width in sorted(by_width):
        rows = by_width[width]

        def med(key, source=rows):
            vals = [r[key] for r in source if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        # W items in fanout_ms against one item's latency at W=1, formed per episode against its own
        # topic's baseline so a run spanning topics cannot produce a ratio between two documents.
        # The W=1 row reads at or just BELOW 1.0 by whatever dispatch and join cost falls outside
        # the item's own span: W=1 prices the machinery, so a gain at W=4 must beat that, not 1.0.
        gains = []
        for r in rows:
            base = baseline.get(r["qid"])
            if base and r["status"] == "ok" and r.get("fanout_ms"):
                gains.append(r["width"] * statistics.median(base) / r["fanout_ms"])

        lines.append(
            f"| {width} | {len(rows)} | {fmt(med('item_ms_p50'), '.1f', 1e-3)} | "
            f"{fmt(med('fanout_ms'), '.1f', 1e-3)} | {fmt(med('peak_overlap'), '.0f')} | "
            f"{fmt(med('makespan_ratio'))} | "
            f"{fmt(statistics.median(gains) if gains else None)} | "
            f"{fmt(med('grounding_rate'), '.0%')} |"
        )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def fmt_s(ms) -> str:
    """Milliseconds as a seconds string, for the per-episode progress line."""
    return "n/a" if ms is None else f"{ms / 1000:.1f}s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=os.environ.get("BENCH_BACKEND", DEFAULT_BACKEND),
                   help="LLM backend: bedrock, openai, anthropic, vllm")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", DEFAULT_MODEL))
    p.add_argument("--harness", default=os.environ.get("BENCH_HARNESS", "native"),
                   help="agent harness to drive (see HARNESSES in this file)")
    p.add_argument("--region", default=os.environ.get("BENCH_REGION", DEFAULT_REGION))
    p.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help=f"per-agent ReAct turn cap (default {DEFAULT_MAX_STEPS}); a runaway "
                        "agent is charged W times over in one episode")
    p.add_argument("--widths", default=os.environ.get("BENCH_WIDTHS", DEFAULT_WIDTHS),
                   help=f"comma-separated fan-out widths (default {DEFAULT_WIDTHS}); keep 1, "
                        "it is the baseline every ratio divides by")
    p.add_argument("--topics", type=int, default=int(os.environ.get("BENCH_TOPICS", DEFAULT_TOPICS)),
                   help=f"how many pinned topics to run (default {DEFAULT_TOPICS}); taken in "
                        "qid STRING order, so 8 means qids 1, 10, 11, 12, ... not 1..8")
    p.add_argument("--topic", action="append", metavar="QID",
                   help="run only this topic id; repeat the flag to select several")
    p.add_argument("--trials", type=int, default=1,
                   help="repeat each (topic, width) N times; the only way to bound baseline noise")
    p.add_argument("--join-mode", choices=JOIN_MODES,
                   default=os.environ.get("BENCH_JOIN_MODE", "dependency"),
                   help="dependency: deref each pending result in item order (what the app "
                        "does). barrier: agsync() the whole wave, opening an agsync:join span")
    p.add_argument("--base-image", default=os.environ.get("BENCH_BASE_IMAGE", "agency-sandbox:latest"))
    p.add_argument("--corpus", type=Path,
                   default=Path(os.environ.get("BENCH_CORPUS", DEFAULT_CORPUS_DIR)),
                   help="the corpus tree written by prepare.py --corpus")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    # Cheapest-first: a missing API key must not surface when the real problem is a bad --widths.
    if args.harness not in HARNESSES:
        sys.exit(f"unknown harness {args.harness!r}; valid: {', '.join(sorted(HARNESSES))}")
    if args.trials < 1:
        sys.exit(f"--trials must be >= 1, got {args.trials}")

    # The whole pin is kept alongside the selection: check_corpus digests every pinned document,
    # and corpus_digest then identifies the same corpus whatever --topics was.
    pinned = load_topics()
    topics = list(pinned)
    if args.topic:
        wanted = set(args.topic)
        unknown = wanted - {t["qid"] for t in topics}
        if unknown:
            sys.exit(f"unknown topic id(s): {', '.join(sorted(unknown))}\n"
                     f"pinned ids are in {DATASET_FILE}")
        topics = [t for t in topics if t["qid"] in wanted]
    else:
        topics = topics[: args.topics]
    if not topics:
        sys.exit("no topics selected")

    # Bounded by the pin, not an arbitrary cap: candidates[:W] IS the wave, so a wider W has no
    # documents to hand its extra agents.
    available = min(len(t["candidates"]) for t in topics)
    widths = []
    for raw in args.widths.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if not raw.isdigit() or not 1 <= int(raw) <= available:
            sys.exit(f"--widths entry {raw!r} must be an integer in 1..{available} "
                     f"(the pinned candidates per topic). Re-pin with "
                     f"`python prepare.py --dataset --force --candidates N` for wider waves.")
        widths.append(int(raw))
    if not widths:
        sys.exit("--widths selected nothing")
    widths = sorted(set(widths))

    corpus = check_corpus(pinned, args.corpus)
    if corpus["manifest_base_image"] and corpus["manifest_base_image"] != args.base_image:
        print(f"  WARNING: the corpus manifest recorded base image "
              f"{corpus['manifest_base_image']!r} but this run uses {args.base_image!r}; "
              "base_image_id in every record will describe the wrong image.")

    # Built last: it reads the API key, so cheap argument and corpus errors surface first.
    llm_config = build_llm_config(args.backend, args.model, args.region, args.context_limit)

    run_dir = args.out / args.harness / args.model.replace("/", "_").replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes = [(t, w, trial) for t in topics for w in widths
                for trial in range(1, args.trials + 1)]
    # Agent runs, not billed calls: every run is a ReAct loop billing once per step, by a multiplier
    # the model chooses and only --max-steps bounds. So the cost line prints runs plus a floor.
    runs = sum(w + 1 for _, w, _ in episodes)
    print(f"harness={args.harness}  backend={args.backend}  model={args.model}")
    print(f"topics={len(topics)}  widths={widths}  trials={args.trials}  "
          f"join_mode={args.join_mode}")
    print(f"corpus={args.corpus}  digest={corpus['digest'][:12]}  fstype={corpus['fstype']}")
    print(f"sandbox={args.base_image}  runtime={corpus['container_runtime'] or 'agency-resolved'}")
    print(f"output={run_dir}")
    # Before the first sandbox: there is no offline mode, and wide sweeps get expensive quietly.
    print(f"\n  COST: {len(episodes)} episodes, {runs} agent runs, {runs} sandboxes.")
    print(f"  Each run is a multi-step ReAct loop, so expect roughly {runs * 3}-{runs * 5} "
          f"billed model calls (observed ~4x; capped at {args.max_steps} steps per run).")
    if 1 not in widths:
        print("  NOTE: no W=1 row in this run, so throughput_gain will read n/a -- there is\n"
              "  nothing to divide by. Add 1 to --widths.")
    print()

    records: list[dict] = []
    interrupted = False
    try:
        for i, (topic, width, trial) in enumerate(episodes, 1):
            print(f"=== [{i}/{len(episodes)}] qid={topic['qid']} W={width} trial={trial} ===")
            ep_out = run_dir / topic["qid"] / f"W{width}" / f"t{trial}"
            rec = run_episode(topic, width, trial, llm_config, args, corpus, ep_out)
            records.append(rec)
            print(f"   status={rec['status']}  peak_overlap={rec.get('peak_overlap')}/{width}  "
                  f"fanout={fmt_s(rec.get('fanout_ms'))}  "
                  f"makespan={rec.get('makespan_ratio')}  {rec['wall_clock_s']}s\n")
            # Rewritten every episode: Ctrl-C should not discard the episodes already paid for.
            write_aggregate(records, run_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- keeping results for completed episodes.")

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"Done: {ok}/{len(records)} episodes completed. Results in {run_dir}")
    if interrupted:
        return 130
    return 0 if (records and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
