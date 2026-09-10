"""
One-time setup for the bug-localization benchmark.

    python prepare.py            # all steps below, in dependency order

    python prepare.py --rg       # fetch the static ripgrep baked into the sandbox images
    python prepare.py --dataset  # select and pin the SWE-bench Verified subset
    python prepare.py --pull     # pull the digest-pinned upstream base images
    python prepare.py --build    # build the per-instance sandbox images (see Dockerfile)
    python prepare.py --check    # verify the built images and the configured model

Every step is idempotent: re-running skips whatever is already done. Add --force to
redo the dataset selection or the image builds.

None of this is measured work -- it all runs before any profiler session opens. Pulling and
building images here rather than lazily is deliberate: the sandbox container starts on its
first exec, inside the session, so an image Docker still has to fetch would charge a
multi-GB download to the `sandbox:start` span and inflate the I/O being measured.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset" / "selected_instances.jsonl"
# The issue text and gold patch, fetched at setup rather than committed: they are verbatim
# upstream dataset content, so the repo pins references to it instead of copying it.
MATERIALIZED_FILE = HERE / "dataset" / "materialized" / "instances.jsonl"

HF_DATASET = "princeton-nlp/SWE-bench_Verified"
HF_SPLIT = "test"
# SWE-bench Verified annotates each instance with an estimated time-to-resolve
# bucket. Verified against the live schema by --dataset, which fails loudly with
# the full column list rather than silently mis-bucketing.
DIFFICULTY_COLUMN = "difficulty"
N_PER_CATEGORY = 3
# Instances are sampled at random within each difficulty bucket, seeded so the selection
# is reproducible. Sorting by instance_id instead would also be deterministic, but
# alphabetically biased: it concentrates the sample in whichever repository sorts first,
# a poor spread for a benchmark whose cost depends on repository size and tree shape.
SELECTION_SEED = 24
# This benchmark stresses storage I/O, so a small repository gives the agent almost
# nothing to navigate. Candidates are restricted to repositories of at least this size
# before sampling. Sizes live in dataset/repo_sizes.json (GitHub API, MB); the relative
# ranking is what matters, not the absolute values. At 100 MB roughly 440 of the 500
# instances remain eligible.
REPO_SIZES_FILE = HERE / "dataset" / "repo_sizes.json"
MIN_REPO_MB = 100.0

# Where the official SWE-bench images check the repo out. Confirmed by --check.
REPO_PATH = "/testbed"

PULL_RETRIES = 3

# Agency's sandboxed grep and glob tools shell out to ripgrep. Agency's own sandbox image
# installs it; SWE-bench images do not ship it. Neither tool reports that as an error:
# grep ends its command in `|| true`, so it returns zero matches, and glob falls back to
# `find -name`, which matches basenames only and so returns nothing for the recursive
# patterns a model naturally writes. Baked into the sandbox image by the Dockerfile.
# A static musl build runs in any Linux image regardless of its glibc version.
RG_VERSION = "14.1.1"
RG_URL = (
    f"https://github.com/BurntSushi/ripgrep/releases/download/{RG_VERSION}"
    f"/ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz"
)
RG_SHA256 = "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e"
RG_CACHE = HERE / ".cache" / "rg"

# Repository for the per-instance sandbox images built from the Dockerfile. Each is the
# upstream SWE-bench image plus that file's two fixes, so the container Agency runs is
# exactly the environment the Dockerfile describes -- nothing is patched in at run time.
DERIVED_IMAGE_REPO = "agency-bench/bug-localization"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command and capture its output, for checks that parse it."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _run_streaming(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command with its output going straight to the terminal.

    Used for image pulls and builds: they move tens of GB and take minutes, and a
    captured-output version looks indistinguishable from a hung process.
    """
    return subprocess.run(cmd, text=True, **kw)


def _ok(msg: str) -> None:
    print(f"  \033[32mPASS\033[0m  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}")


def load_instances() -> list[dict]:
    """Read the pinned instance subset.

    benchmark.py deliberately keeps its own copy of this loader so that running the
    benchmark has no import dependency on the setup script.
    """
    if not DATASET_FILE.is_file():
        sys.exit(f"No pinned dataset at {DATASET_FILE}. Run: python prepare.py --dataset")
    with DATASET_FILE.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _image_for(instance_id: str) -> str:
    """Resolve the official SWE-bench image tag for an instance.

    Prefers the swebench package's own helper so that any future naming change is
    inherited automatically; falls back to the documented rule (lowercase, '__' ->
    '_1776_') when that helper is absent or has moved between versions.
    """
    try:
        from swebench.harness.docker_utils import get_instance_docker_image  # type: ignore

        return get_instance_docker_image(instance_id)
    except Exception:
        pass
    try:
        from swebench.harness.test_spec.test_spec import get_instance_docker_image  # type: ignore

        return get_instance_docker_image(instance_id)
    except Exception:
        pass
    safe = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{safe}:latest"


def _amd64_digest(image: str) -> "str | None":
    """Resolve an image tag to its amd64 manifest digest.

    Pinning by digest rather than tag is what actually makes the measured environment
    reproducible: `:latest` is a mutable pointer that upstream can repoint at any time,
    whereas a digest names exact bytes.
    """
    p = _run(["docker", "manifest", "inspect", image], timeout=180)
    if p.returncode != 0:
        return None
    try:
        idx = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None
    # Multi-arch images publish an index; single-arch ones return the manifest directly.
    for m in idx.get("manifests", []):
        if m.get("platform", {}).get("architecture") == "amd64":
            return m["digest"]
    return None


def _derived_tag(instance_id: str) -> str:
    """Tag of the per-instance sandbox image built from the Dockerfile."""
    return f"{DERIVED_IMAGE_REPO}:{instance_id}"


def _load_repo_sizes() -> dict[str, float]:
    """Repository sizes in MB, used to bias selection toward large trees."""
    if not REPO_SIZES_FILE.is_file():
        sys.exit(
            f"{REPO_SIZES_FILE} is missing. Without it every repository scores 0 MB, the "
            "size filter selects nothing, and the top-up falls back to alphabetical order "
            "-- exactly the bias the seeded sampling exists to avoid. Restore the file, or "
            "pass --min-repo-mb 0 to opt out of the size preference deliberately."
        )
    return json.loads(REPO_SIZES_FILE.read_text()).get("sizes_mb", {})


def ensure_rg(force: bool = False) -> Path:
    """Fetch the pinned static ripgrep that the sandbox images bake in.

    Cached under .cache/ so this is a one-time download, and checksum-verified so that a
    corrupted or substituted binary fails loudly rather than being built into every
    sandbox image.
    """
    if RG_CACHE.is_file() and not force:
        return RG_CACHE

    import hashlib
    import io
    import tarfile
    import urllib.request

    print(f"Fetching ripgrep {RG_VERSION} (static musl build) ...")
    with urllib.request.urlopen(RG_URL, timeout=120) as resp:  # noqa: S310 - pinned https URL
        blob = resp.read()

    digest = hashlib.sha256(blob).hexdigest()
    if digest != RG_SHA256:
        sys.exit(f"ripgrep checksum mismatch\n  expected {RG_SHA256}\n  got      {digest}")

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("/rg")), None)
        if member is None:
            sys.exit("no 'rg' binary inside the ripgrep tarball")
        extracted = tf.extractfile(member)
        if extracted is None:
            sys.exit("could not extract 'rg' from the ripgrep tarball")
        RG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        RG_CACHE.write_bytes(extracted.read())

    RG_CACHE.chmod(0o755)
    print(f"  cached at {RG_CACHE} ({RG_CACHE.stat().st_size / 1e6:.1f} MB)")
    return RG_CACHE


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Validate the three assumptions the whole benchmark rests on."""
    print("Preflight checks\n")
    failures = 0

    # The image to probe: first pinned instance if available, else a known-good one.
    # Tolerate an empty/truncated/malformed pin -- this command's job is to produce
    # readable diagnostics, not a raw traceback.
    image = None
    if DATASET_FILE.is_file():
        try:
            pinned = load_instances()
            image = pinned[0]["image_name"] if pinned else None
        except (json.JSONDecodeError, KeyError, SystemExit) as e:
            print(f"  (could not read pinned dataset: {e}; falling back to a default image)")
    if image is None:
        _fail("no pinned dataset -- run `python prepare.py --dataset` first")
        return 1
    print(f"probe image: {image}\n")

    # 1. Repo checkout path. Load-bearing: agency's sandboxed grep/glob/read default
    #    to /workspace, so a wrong repo path means exploration silently finds nothing.
    print("[1/3] repo checkout path inside the sandbox image")
    if _run(["docker", "image", "inspect", image]).returncode != 0:
        # This tag is built locally by --build, never published, so there is nothing to
        # pull; say what to run instead.
        _fail(f"{image} not built -- run `python prepare.py --build`")
        return 1
    if image:
        probe = _run(["docker", "run", "--rm", image, "bash", "-lc", f"test -d {REPO_PATH} && ls {REPO_PATH} | head -5"])
        if probe.returncode == 0 and probe.stdout.strip():
            _ok(f"{REPO_PATH} exists and is populated")
        else:
            _fail(
                f"{REPO_PATH} missing or empty -- set REPO_PATH in prepare.py, then "
                "re-pin with: python prepare.py --dataset --force  (benchmark.py reads "
                "repo_path from the pinned JSONL, not from a constant)"
            )
            listing = _run(["docker", "run", "--rm", image, "bash", "-lc", "ls /"])
            print(f"       contents of /: {listing.stdout.strip()}")
            failures += 1
    else:
        _fail("skipped -- no usable image")

    # 2. Profiler can see the sandbox container's cgroup. This is what breaks if the
    #    driver runs inside a container -- see the Dockerfile header.
    print("\n[2/3] profiler cgroup visibility for a sandbox container")
    if image is None:
        _fail("skipped (no usable image)")
        failures += 1
    else:
        try:
            from agency import agprof, agSandbox
            from agency.agconfig import agConfig
            from agency.agsandbox import agSandboxConfig

            with tempfile.TemporaryDirectory() as td:
                cfg = agConfig(agSandboxConfig(base_image=image))
                sandbox = agSandbox("preflight", agconfig=cfg)
                try:
                    with agprof.session(Path(td) / "prof"):
                        out, rc = sandbox.exec("ls /")
                        if rc != 0:
                            raise RuntimeError(f"sandbox exec failed: {out}")
                    metrics = agprof.summary_metrics() or {}
                    if metrics:
                        _ok("agprof registered the container cgroup and produced metrics")
                    else:
                        _fail("agprof produced no summary metrics")
                        failures += 1

                    # The search tools return empty results rather than errors when
                    # ripgrep is absent, so assert they actually find something. Nothing
                    # is patched in here: the image under test must already be correct.
                    n_files, _ = sandbox.exec('rg --files --glob "**/*.py" /workspace | wc -l')
                    n_hits, _ = sandbox.exec('rg --no-ignore "import" /workspace | wc -l')
                    if (n_files.strip().isdigit() and int(n_files.strip()) > 0
                            and n_hits.strip().isdigit() and int(n_hits.strip()) > 0):
                        _ok(f"search tools functional ({n_files.strip()} py files, "
                            f"{n_hits.strip()} matches via /workspace -> {REPO_PATH})")
                    else:
                        _fail("search tools return nothing from /workspace -- the sandbox "
                              "image is missing ripgrep or the repo symlink; rebuild with "
                              "`python prepare.py --build --force`")
                        failures += 1
                finally:
                    sandbox.destroy()
        except Exception as e:
            _fail(f"{type(e).__name__}: {e}")
            print("       If this is a cgroup/PID error, you are probably running the driver")
            print("       inside a container. Run natively on the host -- see README.md.")
            failures += 1

    # 3. The selected model is actually reachable.
    print("\n[3/3] model reachable")
    if image is None:
        # Without a usable image this would fall back to agency-sandbox:latest and
        # report a missing-image error as if the model were unreachable.
        _fail("skipped -- no usable image to host the probe sandbox")
        failures += 1
    else:
        try:
            import benchmark

            # Fill any unset option from benchmark.py, so preflight and the run agree.
            backend = args.backend or benchmark.DEFAULT_BACKEND
            model = args.model or benchmark.DEFAULT_MODEL
            region = args.region or benchmark.DEFAULT_REGION
            context_limit = args.context_limit or benchmark.DEFAULT_CONTEXT_LIMIT
            print(f"       {backend} / {model}")
            cfg = benchmark.build_llm_config(backend, model, region, context_limit)
            # Reuse the image from check 1: agskill provisions a sandbox even for a
            # tool-less skill, and agency's default image may not be built here.
            reply = benchmark.probe_model(cfg, base_image=image)
            _ok(f"model responded ({reply!r})")
        except Exception as e:
            _fail(f"{type(e).__name__}: {e}")
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) failed.")
    else:
        print("All checks passed.")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# --dataset
# --------------------------------------------------------------------------
def _materialized_ok() -> bool:
    """True when the issue text is present for every pinned instance."""
    if not (DATASET_FILE.is_file() and MATERIALIZED_FILE.is_file()):
        return False
    with MATERIALIZED_FILE.open() as fh:
        have = {json.loads(line)["instance_id"] for line in fh if line.strip()}
    return {inst["instance_id"] for inst in load_instances()} <= have


def cmd_dataset(args: argparse.Namespace) -> int:
    """Select 3 instances per difficulty bucket and pin them to JSONL."""
    if DATASET_FILE.is_file() and not args.force:
        if _materialized_ok():
            print(f"{DATASET_FILE} already exists -- nothing to do. Use --force to reselect.")
            return 0
        # A fresh checkout has the committed pin but not the upstream text, which is
        # deliberately not in git. Fetch the text for the existing pin rather than
        # reselecting, so the instance set stays exactly as committed.
        from datasets import load_dataset

        print(f"Pin present, fetching issue text for it from {HF_DATASET} [{HF_SPLIT}] ...")
        ds = load_dataset(HF_DATASET, split=HF_SPLIT)
        _materialize({inst["instance_id"] for inst in load_instances()}, ds)
        return 0

    from datasets import load_dataset

    print(f"Loading {HF_DATASET} [{HF_SPLIT}] ...")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    if DIFFICULTY_COLUMN not in ds.column_names:
        print("\nAvailable columns:")
        for name in ds.column_names:
            print(f"  - {name}")
        sys.exit(
            f"\nDIFFICULTY_COLUMN={DIFFICULTY_COLUMN!r} is not a column of this dataset. "
            "Update it in prepare.py to the correct name from the list above."
        )

    # Group by bucket, then sample N at random per bucket under a fixed seed. The
    # candidate list is sorted by instance_id first so the sample depends only on the
    # seed, not on the dataset's row order.
    by_bucket: dict[str, list[dict]] = {}
    for row in ds:
        by_bucket.setdefault(row[DIFFICULTY_COLUMN], []).append(row)

    repo_mb = _load_repo_sizes() if args.min_repo_mb > 0 else {}
    print(f"\nFound {len(by_bucket)} difficulty buckets "
          f"(seed={args.seed}, min repo size {args.min_repo_mb} MB):")
    selected: list[dict] = []
    for bucket in sorted(by_bucket):
        rows = sorted(by_bucket[bucket], key=lambda r: r["instance_id"])

        # Prefer large repos, but never fail a bucket over it: if too few qualify,
        # top up with the largest of the rest and say so.
        big = [r for r in rows if repo_mb.get(r["repo"], 0.0) >= args.min_repo_mb]
        note = ""
        if len(big) < N_PER_CATEGORY:
            rest = sorted(
                (r for r in rows if r not in big),
                key=lambda r: -repo_mb.get(r["repo"], 0.0),
            )
            note = (f"  [only {len(big)} repo(s) >= {args.min_repo_mb} MB; "
                    f"topping up with the largest available]")
            big = big + rest[: N_PER_CATEGORY - len(big)]

        rng = random.Random(f"{args.seed}:{bucket}")
        picked = rng.sample(big, min(N_PER_CATEGORY, len(big)))
        picked.sort(key=lambda r: r["instance_id"])
        print(f"  {bucket!r}: {len(rows)} instances "
              f"({len(big)} eligible), selecting {len(picked)}{note}")
        for row in picked:
            upstream = _image_for(row["instance_id"])
            digest = _amd64_digest(upstream)
            if digest:
                base_image = upstream.rsplit(":", 1)[0] + "@" + digest
            else:
                base_image = upstream
                print(f"    warning: no amd64 digest for {upstream}; pinning by tag only")
            selected.append(
                {
                    "instance_id": row["instance_id"],
                    "difficulty": row[DIFFICULTY_COLUMN],
                    "repo": row["repo"],
                    "base_commit": row["base_commit"],
                    # base_image: exact upstream bytes (digest-pinned where resolvable).
                    # image_name: what benchmark.py hands to agconfig -- built from it.
                    "base_image": base_image,
                    "image_name": _derived_tag(row["instance_id"]),
                    "repo_path": REPO_PATH,
                    "selection_seed": args.seed,
                    "repo_size_mb": repo_mb.get(row["repo"]),
                }
            )

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_FILE.open("w") as fh:
        for rec in selected:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(selected)} instances to {DATASET_FILE}")
    print("Commit this file -- it is the pinned source of truth for the benchmark.")

    _materialize({rec["instance_id"] for rec in selected}, ds)
    return 0


def _materialize(instance_ids: set[str], ds) -> None:
    """Write the upstream issue text and gold patch the run needs.

    Kept out of the committed pin and out of git: this is verbatim upstream dataset
    content, and the repo holds references to data rather than copies of it. The pin
    plus this file is what a run consumes, and the pin alone is enough to rebuild it.
    """
    MATERIALIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MATERIALIZED_FILE.open("w") as fh:
        for row in ds:
            if row["instance_id"] in instance_ids:
                fh.write(
                    json.dumps(
                        {
                            "instance_id": row["instance_id"],
                            "problem_statement": row["problem_statement"],
                            "patch": row["patch"],
                        }
                    )
                    + "\n"
                )
    print(f"Wrote issue text and gold patches to {MATERIALIZED_FILE} (gitignored)")


# --------------------------------------------------------------------------
# --pull
# --------------------------------------------------------------------------
def cmd_pull(args: argparse.Namespace) -> int:
    """Pull every pinned upstream base image, and report the disk footprint."""
    instances = load_instances()
    print(f"Ensuring {len(instances)} instance images are present locally\n")

    failed: list[str] = []
    total_bytes = 0

    for i, inst in enumerate(instances, 1):
        image = inst.get("base_image") or inst["image_name"]
        label = f"[{i}/{len(instances)}] {inst['instance_id']}"

        if _run(["docker", "image", "inspect", image]).returncode == 0:
            print(f"{label}: present")
        else:
            for attempt in range(1, PULL_RETRIES + 1):
                print(f"{label}: pulling (attempt {attempt}/{PULL_RETRIES}) ...")
                if _run_streaming(["docker", "pull", image]).returncode == 0:
                    break
                if attempt < PULL_RETRIES:
                    time.sleep(2**attempt)
            else:
                print(f"{label}: FAILED to pull {image}")
                failed.append(inst["instance_id"])
                continue

        size = _run(["docker", "image", "inspect", "--format", "{{.Size}}", image])
        if size.returncode == 0 and size.stdout.strip().isdigit():
            n = int(size.stdout.strip())
            total_bytes += n
            print(f"        size: {n / 1e9:.2f} GB")

    print(f"\nTotal image footprint: {total_bytes / 1e9:.1f} GB")
    if failed:
        print(f"Failed to pull {len(failed)} image(s): {', '.join(failed)}")
        return 1
    return 0


# --------------------------------------------------------------------------
# --build / --rg
# --------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> int:
    """Build the per-instance sandbox images the benchmark profiles.

    Each is one thin layer on top of a digest-pinned SWE-bench base (see the Dockerfile),
    so the container Agency runs is fully described by a committed file rather than being
    patched at run time.
    """
    ensure_rg()
    instances = load_instances()
    print(f"Building {len(instances)} sandbox images from {HERE / 'Dockerfile'}\n")

    failed = []
    for i, inst in enumerate(instances, 1):
        tag = inst["image_name"]
        base = inst.get("base_image") or _image_for(inst["instance_id"])
        label = f"[{i}/{len(instances)}] {inst['instance_id']}"

        if not args.force and _run(["docker", "image", "inspect", tag]).returncode == 0:
            print(f"{label}: present")
            continue

        print(f"{label}: building {tag}")
        built = _run_streaming(
            ["docker", "build",
             "--build-arg", f"BASE_IMAGE={base}",
             "--build-arg", f"REPO_PATH={inst.get('repo_path', REPO_PATH)}",
             "-t", tag, "-f", str(HERE / "Dockerfile"), str(HERE)],
            timeout=1800,
        )
        if built.returncode != 0:
            print(f"{label}: BUILD FAILED (see the build output above)")
            failed.append(inst["instance_id"])

    if failed:
        print(f"\nFailed to build {len(failed)} image(s): {', '.join(failed)}")
        return 1
    print(f"\nAll {len(instances)} sandbox images ready.")
    return 0


def cmd_rg(args: argparse.Namespace) -> int:
    ensure_rg(force=args.force)
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="validate load-bearing assumptions")
    p.add_argument("--dataset", action="store_true", help="select and pin the instance subset")
    p.add_argument("--pull", action="store_true", help="pull the digest-pinned upstream base images")
    p.add_argument("--rg", action="store_true", help="fetch the static ripgrep baked into the sandbox images")
    p.add_argument("--build", action="store_true", help="build the per-instance sandbox images (see Dockerfile)")
    p.add_argument("--force", action="store_true", help="with --dataset, overwrite the existing pin")
    p.add_argument("--min-repo-mb", type=float, default=MIN_REPO_MB,
                   help=f"only sample instances from repos at least this large "
                        f"(default {MIN_REPO_MB} MB); 0 disables the preference")
    p.add_argument("--seed", type=int, default=SELECTION_SEED,
                   help=f"RNG seed for instance selection (default {SELECTION_SEED})")
    # Left as None and resolved from benchmark.py inside --check. Importing benchmark here
    # would make agency a hard dependency of --dataset/--pull/--build/--rg, none of which
    # need it; resolving there keeps preflight probing exactly what benchmark.py will run.
    p.add_argument("--backend", default=os.environ.get("BENCH_BACKEND"),
                   help="backend to probe in --check (default: benchmark.py's)")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL"),
                   help="model to probe in --check (default: benchmark.py's)")
    p.add_argument("--region", default=os.environ.get("BENCH_REGION"))
    p.add_argument("--context-limit", type=int, default=None)
    args = p.parse_args()

    # No flags: run everything, in dependency order.
    if not (args.check or args.dataset or args.pull or args.rg or args.build):
        args.check = args.dataset = args.pull = args.rg = args.build = True

    rc = 0
    if args.rg:
        rc |= cmd_rg(args)
        print()
    # Dependency order: ripgrep is baked into the images, the pin decides which images to
    # build, and --check validates the built result, so it runs last.
    if args.dataset:
        rc |= cmd_dataset(args)
        print()
    if args.pull:
        rc |= cmd_pull(args)
        print()
    if args.build:
        rc |= cmd_build(args)
        print()
    if args.check:
        check_rc = cmd_check(args)
        if check_rc:
            print("\nPreflight failed -- fix the above before continuing.")
        rc |= check_rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
