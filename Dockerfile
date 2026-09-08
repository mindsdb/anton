# minds-anton-scratchpad: the sandbox pod image (anton + scratchpad boot + cloud_turn entrypoint).
# Consumed by scratchpad-controller (SCRATCHPAD_CONTROLLER__SCRATCHPAD_IMAGE) and Minds.
# The controller execs `python -m anton.cloud_turn` (whole turn) or
# `/usr/local/bin/scratchpad-boot.sh` (single cell) inside a gVisor pod running as UID 1000.
FROM python:3.12-slim

# uv is required at RUNTIME: scratchpad_boot shells out to `uv pip install` for missing packages
# via ANTON_UV_PATH=/usr/local/bin/uv, so uv must be present at exactly that path.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

# The version, supplied by the caller because hatch-vcs cannot derive it here:
# .git is excluded from the build context (see .dockerignore) to keep the image
# lean, so there is no history to describe. The workflow resolves it from tags
# and passes it in (ENG-1796).
#
# This was the constant 2.0.0 until 2026-08-26, which meant every pod that ever
# ran reported the same version. It was not a fallback that fired occasionally —
# it was the only value cloud ever reported, and being a well-formed release
# number it read as a legitimate cohort in any breakdown rather than as a null.
# Whatever replaces it must stay derived; a constant here is undetectable
# downstream.
ARG ANTON_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=${ANTON_VERSION} \
    VIRTUAL_ENV=/opt/anton-venv \
    PATH=/opt/anton-venv/bin:/usr/local/bin:$PATH \
    UV_LINK_MODE=copy

WORKDIR /app
COPY . /app

# Install anton into a venv owned by UID 1000 so the non-root runtime can also `uv pip install`
# missing packages at turn time. `uv sync --frozen` installs the exact versions
# from uv.lock; a bare `uv pip install .` re-resolves at build time, which is
# what silently pulled anthropic 1.x (httpx2) into pods while the lockfile
# still pinned 0.x. boto3 is imported by anton at runtime but not declared as
# a dependency, so add it explicitly (the one unlocked package).
# The empty-ARG guard runs first: with no pretend-version and no .git, hatch-vcs
# fails deep inside the sync with an error that does not mention the build arg.
# Fail here instead, naming the cause — and never fall back to a made-up version,
# which is the failure mode this whole change exists to remove.
RUN test -n "$SETUPTOOLS_SCM_PRETEND_VERSION" \
    || { echo "ERROR: --build-arg ANTON_VERSION is required (ENG-1796)." >&2; exit 1; } \
    && case "$SETUPTOOLS_SCM_PRETEND_VERSION" in 2.0.0*) \
         echo "ERROR: ANTON_VERSION=$SETUPTOOLS_SCM_PRETEND_VERSION is the hatch-vcs" >&2; \
         echo "       fallback, which means the resolver found no tags. Baking it would" >&2; \
         echo "       restore the exact constant this guard exists to remove (ENG-1796)." >&2; \
         exit 1 ;; \
       esac \
    && { test ! -e /app/.git \
         || { echo "ERROR: .git entered the build context (ENG-1796). This image is" >&2; \
              echo "       SINGLE-STAGE, so it would ship the repo's full history to every" >&2; \
              echo "       pod — and the pretend-version hides it: the version stays correct," >&2; \
              echo "       so nothing else would ever fail. Keep .git in .dockerignore." >&2; \
              exit 1; }; } \
    && uv venv "$VIRTUAL_ENV" \
    && UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" uv sync --frozen --no-dev --no-cache \
    && uv pip install --no-cache boto3 \
    && chown -R 1000:1000 "$VIRTUAL_ENV"

# scratchpad-boot.sh: the single-cell entrypoint the controller execs (reads code + delimiter on stdin).
RUN printf '#!/bin/sh\nexec python -m anton.core.backends.scratchpad_boot\n' \
      > /usr/local/bin/scratchpad-boot.sh \
    && chmod 0755 /usr/local/bin/scratchpad-boot.sh

# Non-root, matching the pod securityContext (drop ALL caps, no service-account token, fs_group 1000, gVisor).
RUN useradd -u 1000 -m -s /bin/sh scratchpad
USER 1000

# Run both entrypoints once, as the pod's uid, before this image can be pushed.
# The build is otherwise green for an image no pod can serve a turn with: every
# check up to here proves the image was ASSEMBLED, none proves it RUNS. See
# docker/image_smoke.py for what each check is guarding against.
RUN python /app/docker/image_smoke.py

# The controller always execs an explicit command; this default keeps the image runnable standalone.
CMD ["python", "-m", "anton.cloud_turn"]
