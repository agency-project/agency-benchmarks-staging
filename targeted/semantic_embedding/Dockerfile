# The sandbox image for the semantic embedding benchmark.
#
# This defines the environment the agent's encoding runs in, which is the thing being
# measured -- so everything it needs is baked in here. A container is ready the moment
# it starts and nothing is patched in at run time, where the cost would land inside the
# profiler session.
#
# Built by prepare.py --build, which also names the image after a hash of what goes into it,
# so a run cannot silently use an image built from a different corpus or runner. The COPYs
# below expect `prepare.py --fetch` to have run first; a bare `docker build` in a clean
# checkout fails on a missing path. The two supported bases are the same upstream images
# agency-staging/images/Dockerfile{,.nvidia} build on.

# Digest-pinned, not tagged: a tag is a mutable pointer upstream can repoint, and the
# base decides which BLAS torch links against, which is most of what the CPU numbers
# measure. The CUDA pair that `prepare.py --build --gpu` overrides both ARGs with is not
# yet released; see prepare.py's GPU_OPT_IN.
ARG BASE_IMAGE=python:3.12-slim@sha256:a249c9f47e05708dd367f3fe8ada03cf347390fad66fb8b0518c0ef55ae3cb84
FROM ${BASE_IMAGE}

# Overridden to empty on a base that already ships a tuned torch. Letting pip resolve
# torch from PyPI there would replace NVIDIA's build with a generic wheel, changing both
# GPU support and CPU throughput with nothing in the output to show it happened.
ARG TORCH_INSTALL="torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu"
RUN [ -z "$TORCH_INSTALL" ] || pip install --no-cache-dir ${TORCH_INSTALL}

# --no-deps for sentence-transformers for that same reason: its torch dependency would
# otherwise be resolved from PyPI. The unpinned packages are ones pip treats as already
# satisfied when a base image ships them, so this layer cannot downgrade a tuned build.
#
# sentence-transformers is used rather than plain transformers so pooling comes from each
# encoder's own 1_Pooling/config.json -- bge pools on CLS, MiniLM on the token mean, and a
# hand-maintained table that got one wrong would silently cost recall.
RUN pip install --no-cache-dir transformers==5.16.1 numpy scikit-learn scipy tqdm \
 && pip install --no-cache-dir --no-deps sentence-transformers==6.0.0

# A measured run must not reach the network: a lazy download would be charged to the
# workload, and a silently updated encoder would break comparability with past traces.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY .cache/models          /opt/models
COPY dataset/materialized   /opt/bench/data
COPY dataset/models.json    /opt/bench/models.json
COPY dataset/selection.json /opt/bench/selection.json
COPY sandbox_runner.py      /opt/bench/sandbox_runner.py

# Agency creates and works out of /workspace; the CUDA base ships its own populated one.
RUN rm -rf /workspace && mkdir -p /workspace

# Fail the build rather than the benchmark. This asserts every encoder loads and returns
# its published dimension, every variant is complete and internally consistent, and that
# tail queries really do start past a 512-token window -- which prepare.py could only
# approximate in characters, and which the whole quality gradient depends on.
RUN python /opt/bench/sandbox_runner.py --mode verify
