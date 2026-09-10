"""Semantic embedding benchmark -- targets CPU compute.

The agent is given a corpus of news articles and a retrieval quality target, and must
choose the encoder configuration that reaches the target at the lowest CPU cost. It has
two tools: one reports the corpus's token-length distribution, the other encodes the
corpus and scores retrieval. Both run the pinned runner inside the sandbox, so the
parameters are the agent's and the code is not -- identical parameters mean identical
kernels, and two runs stay comparable.

Documents run to a median of about 800 tokens, well past a 512-token window, so
covering one costs several encoder passes. That is the CPU work this benchmark measures.

The primary artifact is the profiler trace under output/. Whether the target was met,
in how many attempts, is a secondary sanity signal.

Run:
    python prepare.py                    # once: fetch, build, check
    python benchmark.py                  # all three corpus variants
    python benchmark.py --variant mixed  # one variant
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agency import agSandbox, agdata, agent, agerror, agprof, agskill
from agency.agconfig import agConfig
from agency.agllm_backends import (
    agAnthropicBackendConfig,
    agBedrockBackendConfig,
    agOpenAIBackendConfig,
    agVLLMBackendConfig,
)
from agency.agresources import agResourcePoolConfig
from agency.agsandbox import agSandboxConfig
from agency.agskill import agSkillConfig
from agency.agtool import agtool
from agency.tools import todowrite

# The pinned facts, and the loaders for them, belong to prepare.py. Importing in this
# direction is free: prepare.py touches agency only inside cmd_check.
from prepare import (
    GPU_OPT_IN,
    RUNNER_IN_IMAGE,
    gpu_unreleased,
    image_tag,
    load_choices,
    load_selection,
    parse_report,
)

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

# Bumped whenever a recorded field is renamed or changes meaning, so a mixed-vintage
# output/ tree stays interpretable. output/ accumulates across code revisions and is the
# documented input for downstream analysis, which is the one consumer that cannot be
# updated in step with the code.
RECORD_SCHEMA_VERSION = 1

DEFAULT_BACKEND = "bedrock"
DEFAULT_LLM_MODEL = "minimax.minimax-m2.5"
DEFAULT_REGION = "us-east-2"
DEFAULT_CONTEXT_LIMIT = 196000

# Agency's own default is 4096, effectively unbounded for a benchmark.
DEFAULT_MAX_STEPS = 40
# A separate bound, because turns are cheap and encoding is not: six passes over the
# corpus is already tens of minutes of CPU, and a step cap alone would not stop an agent
# from spending hours re-encoding.
DEFAULT_MAX_EMBED_CALLS = 6
# sandbox.exec defaults to 120 s, which a full pass on the larger encoders exceeds.
DEFAULT_EMBED_TIMEOUT_S = 1800
# corpus_stats only tokenizes a sample, so this is generous; it exists so the tool cannot
# hang the run if the container wedges.
CORPUS_STATS_TIMEOUT_S = 600
# Thread count is the bound that actually holds. Agency gates its container CPU quota on a
# cgroup v1 probe that fails on any cgroup v2 host -- and the profiler requires cgroup v2 --
# so the quota is requested but never relied on. Matches Agency's idle_cpus default.
DEFAULT_THREADS = 8

MIN_SEQ_LEN = 64
MAX_BATCH_SIZE = 256
# Below DOCS_PER_VARIANT, so corpus_stats stays a sample rather than a full pass over the
# corpus -- its whole justification is that it is cheap relative to an encoding.
MAX_STATS_SAMPLE = 250
# Encode CPU repeats to within about 1% between identical runs, so an exact-equality
# verdict on "cheapest" would be decided by noise whenever two configurations are close --
# and a careful sweep, which is the behaviour worth rewarding, is what produces near-ties.
# 5% sits comfortably outside the measured spread.
CHEAPEST_TIE_TOLERANCE = 1.05

# Harnesses this benchmark can drive, mapped to the extra agConfig sources that select one.
# Adding an entry is the whole change: --harness validates against these keys and the name
# forms the top level of the output path, so runs stay comparable across harnesses.
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
        return agOpenAIBackendConfig(
            model=model, api_key=load_api_key(), context_limit=context_limit
        )
    if backend == "anthropic":
        return agAnthropicBackendConfig(
            model=model, api_key=load_api_key(), context_limit=context_limit
        )
    if backend == "vllm":
        return agVLLMBackendConfig(
            base_url=os.environ["LLM_BASE_URL"],
            model=model,
            api_key=os.environ.get("LLM_API_KEY", ""),
            context_limit=context_limit,
        )
    raise ValueError(f"unknown backend {backend!r}; valid: bedrock, openai, anthropic, vllm")


# Cheapest first. sorted() would give long/mixed/short, so `--limit 1` would run the most
# expensive variant -- the opposite of what a smoke test wants, and the reverse of the order
# every table in the README uses.
VARIANT_ORDER = ("short", "mixed", "long")


def order_variants(names) -> "list[str]":
    """Variants in increasing cost, with any unrecognised name kept at the end."""
    known = [v for v in VARIANT_ORDER if v in names]
    return known + sorted(n for n in names if n not in VARIANT_ORDER)


def format_choices(choices: dict) -> str:
    """Render the encoder and strategy menus for the system prompt.

    Generated from models.json rather than written out here, so the prompt can never
    offer something the image does not implement.
    """
    rows = [f"ENCODERS:\n  {'name':<11}{'params':>8}{'dim':>6}{'max tokens':>12}"]
    for key, spec in choices["encoders"].items():
        rows.append(
            f"  {key:<11}{spec['params_m']:>7.0f}M{spec['dim']:>6}{spec['max_seq_len']:>12}"
        )
    rows.append("\nSTRATEGIES:")
    rows += [f"  {name:<11} {desc}" for name, desc in choices["strategies"].items()]
    return "\n".join(rows)


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
class RunnerTimeout(RuntimeError):
    """The runner exceeded its timeout and was confirmed killed."""


class ReapFailed(RuntimeError):
    """A timed-out runner could not be confirmed dead, so the variant is unmeasurable.

    Abandoned encoding keeps consuming the container's CPU and is charged to the same
    sandbox_metrics total, so every later attempt in this variant would be measured under
    contention nothing accounts for.

    This cannot be signalled by letting the exception escape: agtool.__call__ wraps every
    tool in `except Exception -> agerror`, so it would arrive at the model as a recoverable
    tool error and the ReAct loop would carry on. Deriving from BaseException instead would
    escape agskill's handler too and leave `result_future` unset, hanging .wait() forever.
    So both tools catch it and set state["reap_failed"]; both then refuse every later call,
    and it decides the variant's status regardless of how the skill itself ends.
    """


def _run_runner(sandbox: agSandbox, argv: list[str], timeout: int) -> dict:
    output, rc = sandbox.exec(" ".join(argv), timeout=timeout)
    if rc != 0:
        # A timeout only kills the host-side `docker exec` client; the runner keeps going
        # inside the container (agsandbox_backends/container.py returns rc=-1 here). Left
        # alone it would saturate the CPU through every later attempt and be charged to
        # the same sandbox_metrics total, silently corrupting the rest of the variant.
        if rc == -1 and "timed out" in output.lower():
            # The kill has to be confirmed, not assumed. Three outcomes are otherwise
            # indistinguishable: killed the runaway, found nothing to kill, or never ran
            # at all -- and only the first makes it safe to continue the variant.
            reap_out, reap_rc = sandbox.exec(
                f"python {RUNNER_IN_IMAGE} --mode reap", timeout=120
            )
            try:
                reap = parse_report(reap_out)
            except (ValueError, KeyError, json.JSONDecodeError):
                raise ReapFailed(
                    f"runner exceeded {timeout}s and the reap did not report "
                    f"(rc={reap_rc}): {reap_out.strip()[-300:]}"
                ) from None
            # `.get()` rather than `[...]`: an image built before the reap reported
            # `remaining` would otherwise raise KeyError, which agtool turns into a plain
            # tool error with no flag set -- walking straight through the abort. A report
            # that cannot confirm the kill counts as a failure to confirm it.
            if reap_rc != 0 or reap.get("remaining") is None or reap["remaining"]:
                raise ReapFailed(
                    f"runner exceeded {timeout}s and the reap did not confirm it died "
                    f"(rc={reap_rc}, remaining={reap.get('remaining')})"
                )
            raise RunnerTimeout(
                f"runner exceeded {timeout}s and was killed (reaped {reap['reaped']})"
            )
        raise RuntimeError(f"runner exited {rc}: {output[-1500:]}")
    return parse_report(output)


def make_corpus_stats_tool(
    sandbox: agSandbox,
    variant: str,
    encoders: dict,
    args: argparse.Namespace,
    state: dict,
) -> agtool:
    """The length signal, measured in a chosen encoder's own tokens.

    Sampled rather than exhaustive: tokenizing the whole corpus would add a visible slice
    of CPU to the run and blur the attribution of the encoding it exists to inform.
    """

    def _run(arg: agdata) -> agdata:
        if state.get("reap_failed"):
            return agerror(
                "variant aborted: a timed-out encoder could not be confirmed dead, so "
                "further measurements here are not interpretable"
            )
        model = getattr(arg, "model", None)
        if model not in encoders:
            return agerror(f"unknown model {model!r}; valid: {', '.join(encoders)}")
        try:
            sample_size = int(getattr(arg, "sample_size", 200) or 200)
        except (TypeError, ValueError) as e:
            return agerror(f"sample_size must be an integer: {e}")
        argv = [
            "python", RUNNER_IN_IMAGE,
            "--mode", "stats",
            "--variant", variant,
            "--model", str(model),
            "--sample-size", str(max(1, min(sample_size, MAX_STATS_SAMPLE))),
            # Pinned like the encode path: without this the tokenizer falls back to the
            # host core count and its CPU lands in the same sandbox_metrics total.
            "--threads", str(args.threads),
        ]
        try:
            return agdata(**_run_runner(sandbox, argv, timeout=CORPUS_STATS_TIMEOUT_S))
        except ReapFailed as e:
            state["reap_failed"] = str(e)
            state["timed_out"] = True
            return agerror(f"{e}. This variant can no longer be measured.")
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            return agerror(f"corpus_stats failed: {e}")

    return agtool(
        name="corpus_stats",
        fn=_run,
        description=(
            "Token-length distribution of the corpus, measured with one encoder's "
            "tokenizer. Reports percentiles, the fraction of documents longer than a "
            "256- and 512-token window, and the average number of chunks per document "
            "at each. Cheap: it samples documents rather than reading all of them."
        ),
        params={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": sorted(encoders),
                    "description": "whose tokenizer to measure with",
                },
                "sample_size": {
                    "type": "integer",
                    "description": f"documents to sample (default 200, max {MAX_STATS_SAMPLE})",
                },
            },
            "required": ["model"],
        },
        run_in_subprocess=False,
    )


def make_embed_corpus_tool(
    sandbox: agSandbox,
    variant: str,
    args: argparse.Namespace,
    state: dict,
    encoders: dict,
    target: float,
) -> agtool:
    """The measured work: encode the corpus, then score retrieval over it."""

    def _run(arg: agdata) -> agdata:
        if state.get("reap_failed"):
            return agerror(
                "variant aborted: a timed-out encoder could not be confirmed dead, so "
                "further measurements here are not interpretable"
            )
        if state["calls"] >= args.max_embed_calls:
            # A budget, not a failure: the agent should now commit to the best
            # configuration it has evidence for rather than keep spending CPU.
            return agerror(
                f"attempt budget exhausted ({args.max_embed_calls} encodings). "
                "Report the best configuration you have measured."
            )

        model = getattr(arg, "model", None)
        strategy = getattr(arg, "strategy", None)
        if model not in encoders:
            return agerror(f"unknown model {model!r}; valid: {', '.join(encoders)}")
        strategies = load_choices()["strategies"]
        if strategy not in strategies:
            return agerror(f"unknown strategy {strategy!r}; valid: {', '.join(strategies)}")

        limit = encoders[model]["max_seq_len"]
        try:
            max_seq_len = int(getattr(arg, "max_seq_len", limit) or limit)
            batch_size = int(getattr(arg, "batch_size", 32) or 32)
        except (TypeError, ValueError) as e:
            return agerror(f"max_seq_len and batch_size must be integers: {e}")
        if not MIN_SEQ_LEN <= max_seq_len <= limit:
            return agerror(
                f"max_seq_len must be between {MIN_SEQ_LEN} and {limit} for {model}"
            )
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            return agerror(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")

        argv = [
            "python", RUNNER_IN_IMAGE,
            "--mode", "embed",
            "--variant", variant,
            "--model", str(model),
            "--max-seq-len", str(max_seq_len),
            "--strategy", str(strategy),
            "--batch-size", str(batch_size),
            "--device", args.device,
            # Explicit rather than left to the runner's cgroup detection, which reads a
            # container quota that is absent on a cgroup v2 host.
            "--threads", str(args.threads),
        ]
        state["calls"] += 1
        try:
            report = _run_runner(sandbox, argv, timeout=args.embed_timeout)
        except ReapFailed as e:
            state["reap_failed"] = str(e)
            state["timed_out"] = True
            return agerror(
                f"{e}. This variant can no longer be measured: report the best "
                "configuration you have already measured and do not encode again."
            )
        except RunnerTimeout as e:
            # Recorded so result.json shows why the numbers stop here.
            state["failures"].append(str(e))
            state["timed_out"] = True
            return agerror(
                f"{e}. That configuration is too expensive for the time budget; "
                "report the best configuration you have already measured."
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            state["failures"].append(str(e))
            return agerror(f"embed_corpus failed: {e}")

        state["attempts"].append(report)
        report["target_recall"] = target
        report["target_met"] = report["recall_at_1"] >= target
        report["attempts_remaining"] = args.max_embed_calls - state["calls"]
        return agdata(**report)

    return agtool(
        name="embed_corpus",
        fn=_run,
        description=(
            "Encode the whole corpus with the given configuration, then score retrieval "
            "and report both quality and cost. Expensive -- a single call can take "
            "minutes of CPU, and the number of calls is capped."
        ),
        params={
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": sorted(encoders)},
                "max_seq_len": {
                    "type": "integer",
                    "description": (
                        f"tokens per sequence, between {MIN_SEQ_LEN} and the encoder's own "
                        "limit (256 for minilm-l6, 512 for the bge encoders)"
                    ),
                },
                "strategy": {
                    "type": "string",
                    "enum": list(load_choices()["strategies"]),
                    "description": "how to handle documents longer than max_seq_len",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "sequences per forward pass (default 32)",
                },
            },
            "required": ["model", "max_seq_len", "strategy"],
        },
        run_in_subprocess=False,
    )


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------
def build_embedding_skill(
    sandbox: agSandbox,
    variant: str,
    args: argparse.Namespace,
    choices: dict,
    state: dict,
    target: float,
) -> agskill:
    """The tuning skill, bound to this variant's sandbox.

    replace_tools is fixed at construction and the tool factories close over a live
    sandbox, so this is rebuilt per variant.
    """
    return agskill(
        name="embedding_tuner",
        system_prompt=(
            "You are tuning a semantic search index over a corpus of news articles.\n\n"
            f"GOAL: find an encoder configuration that reaches recall@1 >= "
            f"{target} at the LOWEST CPU cost. Over-provisioning is a "
            "failure, not a safe choice: if a smaller encoder or a cheaper strategy "
            "also clears the target, the expensive one is the wrong answer.\n\n"
            f"{format_choices(choices)}\n\n"
            "QUERIES come in two kinds, scored separately. 'summary' queries summarize\n"
            "the whole article. 'tail' queries quote a passage from late in it. A\n"
            "configuration that scores well on summary and badly on tail is not weak --\n"
            "it is truncating, and losing the end of every document.\n\n"
            "TOOLS: corpus_stats is cheap, call it first to see how long the documents\n"
            f"are. embed_corpus is expensive and you have at most {args.max_embed_calls} "
            "calls of it, so spend them deliberately.\n\n"
            "Form a hypothesis about the cheapest configuration that could clear the\n"
            "target, test it, and escalate only as far as the evidence requires. Then\n"
            "report the configuration you would ship."
        ),
        replace_tools=[
            make_corpus_stats_tool(sandbox, variant, choices["encoders"], args, state),
            make_embed_corpus_tool(sandbox, variant, args, state, choices["encoders"], target),
            todowrite,
        ],
        input_schema=agdata(variant=str, corpus_docs=int, target_recall=float),
        output_schema=agdata(
            embedding_model=str,
            max_seq_len=int,
            strategy=str,
            batch_size=int,
            target_met=bool,
            reasoning=str,
        ),
    )


# --------------------------------------------------------------------------
# scoring -- secondary signal only
# --------------------------------------------------------------------------
def _encode_cost(attempt: dict) -> float:
    """Total process CPU for the attempt, minus the one-off encoder load.

    Not purely encoding -- tokenization, retrieval and the torch import are in it too --
    but those are near-constant across configurations, so it ranks them fairly.
    """
    return attempt.get("encode_cpu_seconds", attempt["cpu_seconds"])


def _config_of(attempt: dict) -> dict:
    """The three fields that identify a configuration, for comparing attempts."""
    return {k: attempt[k] for k in ("model", "max_seq_len", "strategy")}


def score_run(state: dict, final: dict, target: float) -> dict:
    """Summarize what the agent achieved and what it spent.

    The judgement that matters is `final_is_cheapest_passing`, not recall. The task is to
    clear the target as cheaply as possible, so an agent that reports slightly lower
    recall for much lower cost has done the job well -- scoring against the highest
    recall it happened to measure would mark exactly that behaviour wrong.
    """
    attempts = state["attempts"]
    passing = [a for a in attempts if a["recall_at_1"] >= target]
    # Ranked on encode cost, not total process CPU: including the weight load would let a
    # cold large-encoder load flip a close comparison. See load_cpu_seconds in the runner.
    cheapest = min(passing, key=_encode_cost) if passing else None
    strongest = max(attempts, key=lambda a: a["recall_at_1"]) if attempts else None
    reported = {
        "model": final.get("embedding_model"),
        "max_seq_len": final.get("max_seq_len"),
        "strategy": final.get("strategy"),
    }
    return {
        "attempts": len(attempts),
        "failed_attempts": len(state["failures"]),
        "timed_out": bool(state.get("timed_out")),
        "reap_failed": state.get("reap_failed"),
        "target_recall": target,
        "target_met": bool(passing),
        # The highest recall measured, kept as a diagnostic: it says how much quality was
        # available, which is not the same as what the task rewards.
        "max_recall_at_1": strongest["recall_at_1"] if strongest else None,
        "max_recall_by_kind": strongest["recall_at_1_by_kind"] if strongest else None,
        "cheapest_passing_config": _config_of(cheapest) if cheapest else None,
        "cheapest_passing_recall": cheapest["recall_at_1"] if cheapest else None,
        "cheapest_passing_encode_cpu_seconds": _encode_cost(cheapest) if cheapest else None,
        "reported_config": reported,
        # Credited for any passing configuration within the tie tolerance of the cheapest,
        # so the verdict survives a re-run rather than turning on measurement noise.
        "final_is_cheapest_passing": bool(cheapest)
        and any(
            _config_of(a) == reported
            and _encode_cost(a) <= _encode_cost(cheapest) * CHEAPEST_TIE_TOLERANCE
            for a in passing
        ),
        "reported_target_met": bool(final.get("target_met")),
        # Total process CPU across attempts, weight loads included -- what the run
        # actually spent, as opposed to what configurations are ranked on.
        "encoder_cpu_seconds": round(sum(a["cpu_seconds"] for a in attempts), 1),
        "encoder_encode_cpu_seconds": round(sum(_encode_cost(a) for a in attempts), 1),
        "sequences_encoded": sum(a["sequences_encoded"] for a in attempts),
        "tokens_encoded": sum(a["tokens_encoded"] for a in attempts),
        "peak_rss_mb": max((a["peak_rss_mb"] for a in attempts), default=None),
    }


# --------------------------------------------------------------------------
# run loop
# --------------------------------------------------------------------------
def _finalize(record: dict, state: dict, out_dir: Path, started: float) -> dict:
    """Attach the attempt log and the profiler's sandbox figures, then persist.

    Shared by all three exits -- normal, exception and interrupt -- so result.json always
    has the same shape, and an interrupted variant keeps the attempts it already paid for.
    """
    record["wall_clock_s"] = round(time.perf_counter() - started, 2)
    record["attempt_log"] = list(state["attempts"])

    # The container's CPU figure is the headline metric, so it belongs in the record and
    # the aggregate table, not only in the trace. Unlike the runner's own per-attempt
    # number this covers the whole session, container startup and tool overhead included.
    try:
        summary = json.loads((out_dir / "profiler" / "summary.json").read_text())
        rows = summary.get("sandbox_metrics") or []
        # Selected by label, not by position: agprof sorts these rows alphabetically, so
        # rows[0] is only this sandbox's row because there happens to be exactly one.
        # Anything that ever adds a second would silently swap in another container's CPU
        # total as the headline number.
        row = next(
            (r for r in rows if r.get("label") == record.get("agname")),
            rows[0] if rows else {},
        )
        record["sandbox_cpu_seconds"] = row.get("cpu_time_seconds")
        record["sandbox_cpu_peak_percent"] = row.get("cpu_peak_percent")
        record["sandbox_memory_peak_mb"] = row.get("memory_peak_mb")
    except (OSError, json.JSONDecodeError, IndexError):
        # Same keys as the success path, so downstream analysis never has to distinguish
        # a missing field from a null one.
        record["sandbox_cpu_seconds"] = None
        record["sandbox_cpu_peak_percent"] = None
        record["sandbox_memory_peak_mb"] = None

    (out_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def run_variant(
    variant: str, selection: dict, choices: dict, llm_config, args, out_dir: Path
) -> dict:
    """Run one corpus variant under its own profiler session."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pinned with the corpus, because how hard a given recall@1 is depends on the
    # variant. --target-recall overrides it for every variant at once.
    target = (
        args.target_recall
        if args.target_recall is not None
        else selection["target_recall"][variant]
    )

    # Built before the try so a failure still produces a result record.
    record = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "variant": variant,
        "corpus_docs": selection["docs_per_variant"],
        "target_recall": target,
        "harness": args.harness,
        "backend": args.backend,
        "llm_model": args.model,
        "threads": args.threads,
        "device": args.device,
        "max_steps": args.max_steps,
        "max_embed_calls": args.max_embed_calls,
    }

    # Per-variant log dir. agent.log_dir is process-global, so this is only unambiguous
    # because variants run sequentially -- parallelising them would need it per agent.
    # aglog writes <agname>_history.jsonl (the full LLM
    # conversation) and <agname>_timeline.jsonl here. Keeping them beside the variant's
    # own trace -- rather than pooled per run -- is what makes a recorded run replayable
    # later: the request/response sequence is unambiguously tied to one variant.
    agent.log_dir = out_dir / "agent_logs"

    state = {"calls": 0, "attempts": [], "failures": [], "timed_out": False,
             "reap_failed": None}
    sandbox = None
    started = time.perf_counter()
    try:
        cfg = agConfig(
            llm_config,
            agSandboxConfig(base_image=image_tag()),
            # Requested to match the thread count so the two can never disagree where
            # Agency can apply it; see DEFAULT_THREADS for why it often cannot.
            agResourcePoolConfig(idle_cpus=float(args.threads)),
            agSkillConfig(react_max_steps=args.max_steps),
            *HARNESSES[args.harness](),
        )
        ag = agent(agconfig=cfg)

        # Bind the sandbox before building the skill: both tools close over this exact
        # instance, and agskill only lazily provisions one when ag.sandbox is None.
        record["agname"] = ag.agname  # maps agent_logs/<agname>_*.jsonl to this variant
        sandbox = agSandbox(ag.agname, agconfig=ag.agconfig)
        ag.sandbox = sandbox
        skill = build_embedding_skill(sandbox, variant, args, choices, state, target)

        # The session must stay open until the result resolves: ag.run() is
        # non-blocking, and closing early would truncate the trace.
        with agprof.session(out_dir / "profiler"):
            result = ag.run(
                skill,
                agdata(
                    variant=variant,
                    corpus_docs=selection["docs_per_variant"],
                    target_recall=target,
                ),
            ).wait()

        # An errored run resolves to data carrying an "error" key rather than to an
        # agerror instance: agdata._resolve() copies the future's _data into the pending
        # object without changing its class, so isinstance() would not catch it.
        data = result.to_dict()
        if state.get("reap_failed"):
            # Checked first: the skill will have returned normally or with its own error,
            # but neither describes why this variant's CPU total is uninterpretable.
            record["status"] = "reap_failed"
            record["error"] = state["reap_failed"]
            # Kept: it is what the agent believed when it was cut off, and the main
            # diagnostic left on a variant whose numbers cannot be used.
            record["reasoning"] = data.get("reasoning", "")
            record.update(score_run(state, {}, target))
        elif "error" in data:
            # A capped or failed run is still a valid systems trace -- keep it.
            msg = str(data["error"])
            record["status"] = "max_steps_exceeded" if "max_steps" in msg else "error"
            record["error"] = msg
            record.update(score_run(state, {}, target))
        else:
            record["status"] = "ok"
            record["reasoning"] = data.get("reasoning", "")
            record.update(score_run(state, data, target))
    except KeyboardInterrupt:
        # agprof.session() has already flushed this variant's trace during unwinding and
        # its attempts are already paid for, so record them before the interrupt reaches
        # main() -- otherwise the expensive in-flight variant is the one that is lost.
        #
        # Snapshot first: ag.run() resolves on a daemon thread that Ctrl-C does not stop,
        # so it may still append to state["attempts"] while we score and serialise it.
        snapshot = {
            **state,
            "attempts": list(state["attempts"]),
            "failures": list(state["failures"]),
        }
        record["status"] = "interrupted"
        record.update(score_run(snapshot, {}, target))
        _finalize(record, snapshot, out_dir, started)
        raise
    except Exception as e:
        record["status"] = "exception"
        record["error"] = f"{type(e).__name__}: {e}"
        record.update(score_run(state, {}, target))
    finally:
        if sandbox is not None:  # may be None if construction itself failed
            try:
                sandbox.destroy()
            except Exception as e:  # teardown must not mask the run's own result
                print(f"   warning: sandbox teardown failed: {e}")

    return _finalize(record, state, out_dir, started)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Markdown table lines, with the separator row derived from the headers.

    Written out by hand the separator drifts out of sync with the columns, which is a
    silent formatting break rather than an error.
    """
    return (
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _config_label(config: "dict | None") -> str:
    if not config or not config.get("model"):
        return "n/a"
    return f"{config['model']}/{config['strategy']}@{config['max_seq_len']}"


def _num(value, fmt: str = ".0f") -> str:
    return "n/a" if value is None else format(value, fmt)


def write_aggregate(records: list[dict], out_dir: Path) -> None:
    (out_dir / "results.json").write_text(json.dumps(records, indent=2))

    summary = _md_table(
        ["Variant", "Status", "Attempts", "Target", "Best measured recall@1", "Met",
         "Chosen config", "Cheapest passing", "Encode CPU (s)", "Total CPU (s)",
         "Sandbox CPU (s)", "Wall (s)"],
        [[
            r["variant"], r["status"], str(r.get("attempts", 0)),
            _num(r.get("target_recall"), ".2f"), _num(r.get("max_recall_at_1"), ".3f"),
            "yes" if r.get("target_met") else "no",
            f"`{_config_label(r.get('reported_config'))}`",
            # n/a, not "no", when nothing passed: there was no cheaper configuration to
            # have preferred.
            "n/a" if not r.get("target_met")
            else ("yes" if r.get("final_is_cheapest_passing") else "no"),
            _num(r.get("encoder_encode_cpu_seconds")), _num(r.get("encoder_cpu_seconds")),
            _num(r.get("sandbox_cpu_seconds")), _num(r.get("wall_clock_s")),
        ] for r in records],
    )

    attempts = _md_table(
        ["Variant", "#", "Config", "recall@1", "summary", "tail", "Seqs", "Tokens",
         "Encode CPU (s)", "Load CPU (s)", "Tokens/s"],
        [[
            r["variant"], str(i), f"`{_config_label(a)}`", _num(a["recall_at_1"], ".3f"),
            _num(a["recall_at_1_by_kind"].get("summary"), ".3f"),
            _num(a["recall_at_1_by_kind"].get("tail"), ".3f"),
            str(a["sequences_encoded"]), str(a["tokens_encoded"]),
            _num(_encode_cost(a)), _num(a.get("load_cpu_seconds"), ".1f"),
            str(a["tokens_per_s"]),
        ] for r in records for i, a in enumerate(r.get("attempt_log", []), 1)],
    )

    (out_dir / "results.md").write_text("\n".join(
        ["# Semantic embedding results", "",
         "Profiler traces are the primary artifact; these are a secondary sanity signal.", ""]
        + summary
        + ["", "## Attempts", "",
           "Ranked on encode CPU, which excludes the one-off encoder load.", ""]
        + attempts
    ) + "\n")


def main() -> int:
    selection = load_selection()
    variant_names = order_variants(selection["variants"])

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", default=os.environ.get("BENCH_BACKEND", DEFAULT_BACKEND),
                   help="LLM backend: bedrock, openai, anthropic, vllm")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", DEFAULT_LLM_MODEL),
                   help="the driving LLM, not the encoder (the agent chooses that)")
    p.add_argument("--harness", default=os.environ.get("BENCH_HARNESS", "native"),
                   help="agent harness to drive (see HARNESSES in this file)")
    p.add_argument("--region", default=os.environ.get("BENCH_REGION", DEFAULT_REGION))
    p.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    p.add_argument("--variant", action="append", metavar="NAME",
                   help=f"run only this corpus variant ({', '.join(variant_names)}); "
                        "repeat the flag to select several")
    p.add_argument("--limit", type=int, default=None, help="only run the first N variants")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help=f"encoder threads, and the CPU quota requested for the container "
                        f"(default {DEFAULT_THREADS}); the scaling sweep varies this")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="NOT YET RELEASED: cuda requires an image built with "
                        f"prepare.py --build --gpu; set {GPU_OPT_IN}=1 to try it")
    p.add_argument("--target-recall", type=float, default=None,
                   help="override the per-variant target pinned with the corpus "
                        f"({', '.join(f'{k}={v}' for k, v in sorted(selection['target_recall'].items()))})")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help=f"per-variant ReAct turn cap (default {DEFAULT_MAX_STEPS}; "
                        "agency's own default of 4096 is effectively unbounded)")
    p.add_argument("--max-embed-calls", type=int, default=DEFAULT_MAX_EMBED_CALLS,
                   help=f"encodings allowed per variant (default {DEFAULT_MAX_EMBED_CALLS})")
    p.add_argument("--embed-timeout", type=int, default=DEFAULT_EMBED_TIMEOUT_S,
                   help=f"seconds one encoding may take (default {DEFAULT_EMBED_TIMEOUT_S})")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    if args.harness not in HARNESSES:
        sys.exit(f"unknown harness {args.harness!r}; valid: {', '.join(sorted(HARNESSES))}")
    if args.threads < 1:
        sys.exit(f"--threads must be >= 1, got {args.threads}")
    if args.device == "cuda" and not os.environ.get(GPU_OPT_IN):
        return gpu_unreleased("--device cuda")
    if args.device == "cuda":
        # The prompt's objective and the cheapest-passing ranking are both CPU-time based,
        # so on GPU they rank host-side overhead rather than the encoding itself. This is
        # part of why the path is unreleased: the metrics need rethinking, not just a run.
        print("  NOTE: --device cuda moves the encoding off the CPU, so the cost objective\n"
              "  and cheapest_passing ranking no longer describe the measured work.\n")
    if args.max_embed_calls < 1:
        sys.exit(f"--max-embed-calls must be >= 1, got {args.max_embed_calls}")

    variants = variant_names
    if args.variant:
        unknown = set(args.variant) - set(variant_names)
        if unknown:
            sys.exit(f"unknown variant(s): {', '.join(sorted(unknown))}\n"
                     f"pinned variants are {', '.join(variant_names)}")
        variants = [v for v in variant_names if v in set(args.variant)]
    if args.limit is not None:
        variants = variants[: args.limit]

    choices = load_choices()

    # Checked up front: the tag encodes the corpus, encoder table and runner, so a missing
    # image means those changed since the last build rather than that Docker is broken.
    tag = image_tag()
    if subprocess.run(["docker", "image", "inspect", tag],
                      capture_output=True).returncode != 0:
        sys.exit(
            f"No image for the current inputs ({tag}).\n"
            "The corpus, encoder table or runner changed since the last build.\n"
            "Run: python prepare.py --build"
        )
    # Built last: it reads the API key, so cheap argument errors should surface first
    # rather than being masked by a missing-credentials failure.
    llm_config = build_llm_config(args.backend, args.model, args.region, args.context_limit)

    # Keyed by harness/model so sweeps produce siblings instead of clobbering.
    run_dir = args.out / args.harness / args.model.replace("/", "_").replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"harness={args.harness}  backend={args.backend}  llm={args.model}")
    print(f"variants={len(variants)}  threads={args.threads}  device={args.device}")
    targets = {
        v: args.target_recall if args.target_recall is not None else selection["target_recall"][v]
        for v in variants
    }
    print(f"targets recall@1={targets}  max_embed_calls={args.max_embed_calls}")
    print(f"output={run_dir}\n")

    records = []
    interrupted = False
    try:
        for i, variant in enumerate(variants, 1):
            print(f"=== [{i}/{len(variants)}] {variant} ===")
            rec = run_variant(
                variant, selection, choices, llm_config, args, run_dir / variant
            )
            records.append(rec)
            best = rec.get("max_recall_at_1")
            print(
                f"   status={rec['status']}  attempts={rec.get('attempts', 0)}  "
                f"max recall@1={'n/a' if best is None else format(best, '.3f')}  "
                f"target {'met' if rec.get('target_met') else 'MISSED'}"
                # Only meaningful when something passed; with nothing passing there is
                # no cheaper configuration to have preferred.
                f"{'' if not rec.get('target_met') or rec.get('final_is_cheapest_passing') else ' (not the cheapest passing)'}"
                f"  {rec['wall_clock_s']}s\n"
            )
            # Rewrite after every variant: a full sweep is long, and Ctrl-C or a crash
            # should not discard the variants already paid for.
            write_aggregate(records, run_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted -- keeping results for completed variants.")

    ok = sum(1 for r in records if r["status"] == "ok")
    unmeasurable = [r["variant"] for r in records if r["status"] == "reap_failed"]
    print(f"Done: {ok}/{len(records)} completed. Results in {run_dir}")
    if unmeasurable:
        # Non-zero even when other variants succeeded: a scripted sweep should not read
        # a run containing uninterpretable CPU figures as a clean one.
        print(f"  WARNING: unmeasurable variant(s): {', '.join(unmeasurable)}")
    if interrupted:
        return 130
    if unmeasurable:
        return 2
    return 0 if (records and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
