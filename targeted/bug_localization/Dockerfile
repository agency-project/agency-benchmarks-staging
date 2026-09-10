# The sandbox image the agent runs in -- this is the environment being profiled.
#
# Built once per SWE-bench instance by `prepare.py --build`, which passes BASE_IMAGE as a
# digest-pinned reference so the result is byte-identical across machines and immune to
# upstream retagging. benchmark.py then hands the resulting tag to Agency:
#
#     agConfig(agSandboxConfig(base_image="agency-bench/bug-localization:<instance_id>"))
#
# Everything the run needs is baked in here rather than applied to a live container at
# run time, so `docker run` yields a ready environment and the measured container is
# exactly what this file describes.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# 1. ripgrep. Agency's sandboxed grep and glob tools shell out to `rg`. Agency's own
#    sandbox image installs it; SWE-bench images do not ship it, and NEITHER TOOL REPORTS
#    THAT AS AN ERROR. grep ends its command in `|| true`, so a missing `rg` yields zero
#    matches. glob falls back to `find -name`, which matches basenames only, so recursive
#    patterns such as '**/*.py' return nothing. Either way the agent concludes the
#    repository has no matches. A static musl build runs in any Linux image regardless of
#    its glibc version; `prepare.py --rg` fetches it and verifies its checksum.
COPY .cache/rg /usr/local/bin/rg
RUN chmod +x /usr/local/bin/rg && rg --version > /dev/null

# 2. /workspace -> the repository. grep and glob default their `path` argument to
#    /workspace, and Agency's container backend always creates that directory, so in a
#    SWE-bench image it exists but is empty while the repository sits at /testbed. A tool
#    call that omits `path` would therefore search an empty directory and report nothing,
#    again with no error.
ARG REPO_PATH=/testbed
RUN rm -rf /workspace && ln -sfn "${REPO_PATH}" /workspace

# Fail the build rather than a benchmark run if either fix did not take.
RUN test -n "$(rg --files --glob '**/*.py' /workspace | head -1)" \
 || (echo "search tools return nothing from /workspace -> ${REPO_PATH}" && exit 1)
