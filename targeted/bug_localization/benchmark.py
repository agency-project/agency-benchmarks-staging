"""
Bug localization benchmark -- targets storage I/O.

The agent is given a SWE-bench Verified issue and a real repository checked out at the
buggy commit, and must report which files need to change. It has read-only exploration
tools only (grep, glob, read): no edit, no write, no bash. The issue never names the
target files, so the only route to an answer is navigating the tree -- which is the disk
I/O this benchmark exists to measure.

The primary artifact is the profiler trace under output/. File-level precision and recall
against the gold patch are computed as a sanity signal, not as the headline result.

Run:
    python prepare.py            # once: fetch, pin, pull, build, check
    python benchmark.py          # all pinned instances
    python benchmark.py --limit 1 # a single instance
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agency import agSandbox, agdata, agent, agprof, agskill
from agency.agconfig import agConfig
from agency.agllm_backends import (
    agAnthropicBackendConfig,
    agBedrockBackendConfig,
    agOpenAIBackendConfig,
    agVLLMBackendConfig,
)
from agency.agsandbox import agSandboxConfig
from agency.agskill import agSkillConfig
from agency.tools import make_glob, make_grep, make_read, todowrite

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset" / "selected_instances.jsonl"
# Issue text and gold patches, produced by prepare.py --dataset. Not committed: verbatim
# upstream dataset content stays out of the repo, which pins a reference to it instead.
MATERIALIZED_FILE = HERE / "dataset" / "materialized" / "instances.jsonl"
OUTPUT_DIR = HERE / "output"

DEFAULT_BACKEND = "bedrock"
DEFAULT_MODEL = "minimax.minimax-m2.5"
DEFAULT_REGION = "us-east-2"
DEFAULT_CONTEXT_LIMIT = 196000
# Agency's own default is 4096, effectively unbounded for a benchmark: a confused agent
# would keep searching for hours and burn the API budget. 60 turns is ample for
# repository navigation, and a run that exceeds it is still recorded with its trace.
DEFAULT_MAX_STEPS = 60

# Agent harnesses this benchmark can drive. "native" is Agency's own in-process ReAct
# loop, which needs no extra configuration.
#
# To add a harness, map its name to the extra agConfig sources that select it, e.g.
#
#     HARNESSES = {
#         "native": lambda: [],
#         "claude-code": lambda: [agHarnessConfig(harness="claude-code")],
#     }
#
# Nothing else has to change: --harness validates against these keys, and the harness
# name is recorded in every result and forms the top level of the output path, so runs
# under different harnesses land side by side and stay directly comparable.
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
    """Return the Agency backend config view for *backend*.

    Benchmarks in this suite are model- and backend-swappable by convention; nothing here
    hardcodes a provider.
    """
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


def probe_model(llm_config, base_image: str | None = None) -> str:
    """One trivial call, to confirm the model is reachable. Used by prepare.py --check.

    agskill provisions a sandbox even for a tool-less skill, so pass *base_image* to
    reuse an image that is known to exist locally. Otherwise this falls back to
    agency's default (agency-sandbox:latest) and fails on a missing image rather
    than telling you anything about the model.
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

    # Keep probe logs out of Agency's default /tmp location; the caller may not have set
    # a log dir yet, and a probe should not leave artifacts outside the benchmark tree.
    if agent.log_dir is None:
        agent.log_dir = Path(tempfile.mkdtemp(prefix="agency-bench-probe-"))

    ag = agent(agconfig=agConfig(*sources))
    try:
        data = ag.run(skill, agdata(ping="ping")).wait().to_dict()
    finally:
        # agskill lazily provisions a sandbox even for a tool-less skill; without
        # this the container outlives the probe for the rest of the process.
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
def build_localizer_skill(sandbox: agSandbox, repo_path: str) -> agskill:
    """Read-only localization skill, bound to this instance's sandbox.

    replace_tools is fixed at construction and the tool factories need a live
    sandbox, so this is rebuilt per instance against a sandbox that has already
    been assigned to the agent.
    """
    return agskill(
        name="bug_localizer",
        system_prompt=(
            "You are an expert software engineer performing BUG LOCALIZATION ONLY.\n\n"
            "You will NOT fix the bug and you cannot modify anything: your only tools are "
            "read-only exploration (grep, glob, read).\n\n"
            f"The repository is checked out at {repo_path}, already at the commit where the "
            "bug is present. Read the problem statement, then explore the repository until "
            "you are confident which source files must be changed to fix it. The issue will "
            "not name them; you have to find them.\n\n"
            "Then report those file paths RELATIVE to the repository root "
            "(e.g. 'django/db/models/query.py'), most likely first. Report source files that "
            "need to change, not test files."
        ),
        replace_tools=[make_grep(sandbox), make_glob(sandbox), make_read(sandbox), todowrite],
        input_schema=agdata(problem_statement=str),
        output_schema=agdata(relevant_files=list[str], reasoning=str),
    )


# --------------------------------------------------------------------------
# scoring -- secondary signal only
# --------------------------------------------------------------------------
_DIFF_RE = re.compile(r'^diff --git (?:"a/(.+?)"|a/(\S+)) ', re.MULTILINE)


def parse_gold_files(patch: str) -> set[str]:
    """Files touched by the gold patch.

    Handles git's quoted-path form (used when a path contains spaces or
    non-ASCII), which the plain form of this regex silently misses -- and a
    missed gold set would score a correct answer as a total miss.
    """
    return {m[0] or m[1] for m in _DIFF_RE.findall(patch)}


def _normalize(path: str, repo_path: str) -> str:
    """Reduce a path to repo-root-relative form.

    Strips at most ONE prefix. Stripping repeatedly would eat legitimate
    directories: 'a/b/c.py' is a real path, not a diff-prefixed 'b/c.py'.
    Note leading './' is removed as a prefix, not as a character set --
    lstrip('./') would turn '.github/x.yml' into 'github/x.yml'.
    """
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    for prefix in (repo_path.rstrip("/") + "/", repo_path.strip("/") + "/", "a/", "b/"):
        if prefix != "/" and p.startswith(prefix):
            return p[len(prefix) :]
    return p


def score_instance(inst: dict, reported: list[str]) -> dict:
    # Normalize both sides: gold paths come straight from the diff, so a repo
    # whose patch uses 'a/'-prefixed or './'-relative paths would otherwise
    # never intersect with the agent's normalized answers.
    # Filter AFTER normalizing: '.', './' and '/' are truthy inputs but name no file,
    # and left in they would pad the precision denominator.
    _junk = {"", ".", "..", "/"}
    gold = {_normalize(f, inst["repo_path"]) for f in parse_gold_files(inst["patch"])} - _junk
    got = {_normalize(f, inst["repo_path"]) for f in reported} - _junk
    hits = gold & got
    precision = len(hits) / len(got) if got else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "gold_files": sorted(gold),
        "reported_files": sorted(got),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit": bool(hits),
        # An unparseable patch yields an empty gold set, which would otherwise
        # look identical to the agent getting everything wrong.
        "scoring_error": None if gold else "could not parse gold files from patch",
    }


# --------------------------------------------------------------------------
# run loop
# --------------------------------------------------------------------------
def load_instances() -> list[dict]:
    """Join the committed pin with the issue text fetched at setup.

    The pin carries identifiers and image tags; the problem statement and gold patch come
    from MATERIALIZED_FILE, so no upstream prose is committed to this repo.
    """
    if not DATASET_FILE.is_file():
        sys.exit(f"No pinned dataset at {DATASET_FILE}. Run: python prepare.py --dataset")
    with DATASET_FILE.open() as fh:
        instances = [json.loads(line) for line in fh if line.strip()]

    if not MATERIALIZED_FILE.is_file():
        sys.exit(
            f"No issue text at {MATERIALIZED_FILE}.\n"
            "It is fetched at setup rather than committed. Run: python prepare.py --dataset"
        )
    with MATERIALIZED_FILE.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    text = {row["instance_id"]: row for row in rows}

    missing = [i["instance_id"] for i in instances if i["instance_id"] not in text]
    if missing:
        sys.exit(
            f"issue text missing for {len(missing)} pinned instance(s): {', '.join(missing[:3])}"
            f"{' ...' if len(missing) > 3 else ''}\n"
            "Re-fetch it with: python prepare.py --dataset"
        )
    # Copy only the two text fields rather than merging the whole row: the pin is the
    # source of truth for everything else, and a blanket update would let the
    # regenerable file silently override a committed image tag or commit hash.
    for inst in instances:
        row = text[inst["instance_id"]]
        inst["problem_statement"] = row["problem_statement"]
        inst["patch"] = row["patch"]
    return instances


def drop_page_cache() -> bool:
    """Evict the host page cache so the next repo read reaches the block device.

    This is the ONLY way to make `io_read_mb` non-zero, and it is worth being precise
    about why. The repo lives in the instance image's overlayfs lower layers, which the
    host already cached when the image was pulled -- charged to the pull, not to the
    sandbox. Capping the sandbox's memory therefore does not evict them, and the reads
    are served from RAM without ever reaching the device, so io_read_mb reads 0.

    On by default, since the disk-read metric is meaningless without it; --no-drop-caches
    opts out. Host-wide and needs passwordless sudo: it drops cached pages for every
    process on the machine, so on a shared host it is worth coordinating first. Only
    clean pages are discarded (sync runs first), so nothing is lost -- the effect is a
    transient slowdown until caches refill.
    """
    try:
        subprocess.run(["sync"], check=True, timeout=120)
        subprocess.run(
            ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
            input=b"3", check=True, capture_output=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"   warning: could not drop page cache ({e}); io_read_mb will read 0")
        return False


def run_instance(inst: dict, llm_config, args, inst_out: Path) -> dict:
    """Run one instance under its own profiler session."""
    inst_out.mkdir(parents=True, exist_ok=True)

    # Built before the try so a failure still produces a result record. Agent and
    # sandbox construction stay inside the try: without that, one bad instance would
    # abort the loop and lose the aggregate for everything already run.
    record = {
        "instance_id": inst["instance_id"],
        "difficulty": inst["difficulty"],
        "repo": inst["repo"],
        "harness": args.harness,
        "backend": args.backend,
        "model": args.model,
        "max_steps": args.max_steps,
    }

    # Per-instance log dir. aglog writes <agname>_history.jsonl (the full LLM
    # conversation) and <agname>_timeline.jsonl here. Keeping them beside the instance's
    # trace -- rather than pooled per run -- is what makes a recorded run replayable
    # later: the request/response sequence is unambiguously tied to one instance.
    agent.log_dir = inst_out / "agent_logs"

    sandbox = None
    started = time.perf_counter()
    try:
        cfg = agConfig(
            llm_config,
            agSandboxConfig(base_image=inst["image_name"]),
            agSkillConfig(react_max_steps=args.max_steps),
            *HARNESSES[args.harness](),
        )
        ag = agent(agconfig=cfg)

        # Bind the sandbox before building the skill: the read-only tools close over
        # this exact instance, and agskill only lazily provisions one when
        # ag.sandbox is None.
        record["agname"] = ag.agname  # maps agent_logs/<agname>_*.jsonl to this instance
        sandbox = agSandbox(ag.agname, agconfig=ag.agconfig)
        ag.sandbox = sandbox
        skill = build_localizer_skill(sandbox, inst["repo_path"])

        # No runtime patching: ripgrep and the /workspace -> repo symlink are baked into
        # the image by the Dockerfile, so the container is ready as it starts.

        # Outside the profiler session: the eviction itself is setup, not measured work.
        if args.drop_caches:
            record["dropped_caches"] = drop_page_cache()
        # The session must stay open until the result resolves: ag.run() is
        # non-blocking, and closing early would truncate the trace.
        with agprof.session(inst_out / "profiler"):
            result = ag.run(skill, agdata(problem_statement=inst["problem_statement"])).wait()

        # An errored run resolves to data carrying an "error" key rather than to an
        # agerror instance: agdata._resolve() copies the future's _data into the
        # pending object without changing its class, so isinstance() would not
        # catch it. This mirrors agency's own check in agskill.py.
        data = result.to_dict()
        if "error" in data:
            # A capped or failed run is still a valid systems trace -- keep it.
            msg = str(data["error"])
            record["status"] = "max_steps_exceeded" if "max_steps" in msg else "error"
            record["error"] = msg
            record.update(score_instance(inst, []))
        else:
            record["status"] = "ok"
            record["reasoning"] = data.get("reasoning", "")
            record.update(score_instance(inst, data.get("relevant_files") or []))
    except Exception as e:
        record["status"] = "exception"
        record["error"] = f"{type(e).__name__}: {e}"
        # Scoring reads inst["patch"]/inst["repo_path"]; if the failure above was a
        # malformed instance record, scoring here would raise the same error out of
        # the handler and skip the result.json write below.
        try:
            record.update(score_instance(inst, []))
        except Exception as score_exc:
            record["scoring_error"] = f"{type(score_exc).__name__}: {score_exc}"
    finally:
        record["wall_clock_s"] = round(time.perf_counter() - started, 2)
        if sandbox is not None:  # may be None if construction itself failed
            try:
                sandbox.destroy()
            except Exception as e:  # teardown must not mask the run's own result
                print(f"   warning: sandbox teardown failed: {e}")

    # Pull the container's disk-read figure out of the profiler summary: it is the
    # headline metric, so it belongs in the result record and the aggregate table, not
    # only in the trace.
    try:
        summary = json.loads((inst_out / "profiler" / "summary.json").read_text())
        sandbox_metrics = summary.get("sandbox_metrics") or [{}]
        record["io_read_mb"] = sandbox_metrics[0].get("io_read_mb")
        record["sandbox_cpu_seconds"] = sandbox_metrics[0].get("cpu_time_seconds")
    except (OSError, json.JSONDecodeError, IndexError):
        record["io_read_mb"] = None

    (inst_out / "result.json").write_text(json.dumps(record, indent=2))
    return record


def write_aggregate(records: list[dict], out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(records, indent=2))

    lines = [
        "# Bug localization results",
        "",
        "Profiler traces are the primary artifact; these scores are a secondary sanity signal.",
        "",
        "| Instance | Difficulty | Status | P | R | F1 | Hit | Wall (s) | Disk read (MB) | Cold cache |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        io_read = r.get("io_read_mb")
        lines.append(
            f"| `{r['instance_id']}` | {r['difficulty']} | {r['status']} | "
            f"{r.get('precision', 0):.2f} | {r.get('recall', 0):.2f} | {r.get('f1', 0):.2f} | "
            f"{'yes' if r.get('hit') else 'no'} | {r.get('wall_clock_s', 0):.1f} | "
            f"{'n/a' if io_read is None else format(io_read, '.1f')} | "
            f"{'yes' if r.get('dropped_caches') else 'no'} |"
        )

    by_difficulty: dict[str, list[dict]] = {}
    for r in records:
        by_difficulty.setdefault(r["difficulty"], []).append(r)

    lines += ["", "## By difficulty", "", "| Difficulty | N | Mean F1 | Hit rate | Mean wall (s) |", "|---|---|---|---|---|"]
    for bucket in sorted(by_difficulty):
        rows = by_difficulty[bucket]
        n = len(rows)
        lines.append(
            f"| {bucket} | {n} | {sum(r.get('f1', 0) for r in rows) / n:.2f} | "
            f"{sum(1 for r in rows if r.get('hit')) / n:.0%} | "
            f"{sum(r.get('wall_clock_s', 0) for r in rows) / n:.1f} |"
        )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


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
                   help=f"per-instance ReAct turn cap (default {DEFAULT_MAX_STEPS}; agency's own default of 4096 is effectively unbounded)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N instances")
    p.add_argument("--instance", action="append", metavar="INSTANCE_ID",
                   help="run only this instance id; repeat the flag to select several")
    p.add_argument("--drop-caches", action="store_true", dest="drop_caches",
                   help="evict the host page cache before each instance (the default)")
    p.add_argument("--no-drop-caches", action="store_false", dest="drop_caches",
                   help="do NOT evict the host page cache before each instance. Faster and "
                        "leaves other users' caches alone, but io_read_mb will read 0")
    p.set_defaults(drop_caches=True)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    if args.harness not in HARNESSES:
        sys.exit(f"unknown harness {args.harness!r}; valid: {', '.join(sorted(HARNESSES))}")

    if args.limit is not None and args.limit < 0:
        sys.exit(f"--limit must be >= 0, got {args.limit}")

    instances = load_instances()
    if args.instance:
        wanted = set(args.instance)
        unknown = wanted - {i["instance_id"] for i in instances}
        if unknown:
            sys.exit(f"unknown instance id(s): {', '.join(sorted(unknown))}\n"
                     f"pinned ids are in {DATASET_FILE}")
        instances = [i for i in instances if i["instance_id"] in wanted]
    if args.limit is not None:
        instances = instances[: args.limit]

    # Built last: it reads the API key, so cheap argument and dataset errors should
    # surface first rather than being masked by a missing-credentials failure.
    llm_config = build_llm_config(args.backend, args.model, args.region, args.context_limit)

    # Keyed by harness/model so sweeps produce siblings instead of clobbering.
    run_dir = args.out / args.harness / args.model.replace("/", "_").replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"harness={args.harness}  backend={args.backend}  model={args.model}")
    print(f"instances={len(instances)}  max_steps={args.max_steps}")
    print(f"output={run_dir}")
    if args.drop_caches:
        # On by default because io_read_mb is structurally 0 without it, but it is
        # host-wide, so say so plainly rather than doing it silently.
        print("\n  NOTE: dropping the host page cache before each instance so disk-read\n"
              "  numbers are real. This affects every process on this machine (a brief\n"
              "  slowdown while caches refill; nothing is lost). Disable with --no-drop-caches.")
    else:
        print("\n  NOTE: --no-drop-caches set; io_read_mb will read 0 (page-cache hits are\n"
              "  not device reads). Other metrics are unaffected.")
    print()

    records = []
    interrupted = False
    try:
        for i, inst in enumerate(instances, 1):
            print(f"=== [{i}/{len(instances)}] {inst['instance_id']} ({inst['difficulty']}) ===")
            rec = run_instance(inst, llm_config, args, run_dir / inst["instance_id"])
            records.append(rec)
            print(f"   status={rec['status']}  F1={rec.get('f1', 0):.2f}  {rec['wall_clock_s']}s\n")
            # Rewrite after every instance: a full sweep is long, and Ctrl-C or a
            # crash should not discard the instances already paid for.
            write_aggregate(records, run_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- keeping results for completed instances.")

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"Done: {ok}/{len(records)} completed. Results in {run_dir}")
    # Non-zero when nothing succeeded, so CI can tell a clean run from a total failure.
    if interrupted:
        return 130
    return 0 if (records and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
