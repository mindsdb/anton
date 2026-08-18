# minds-anton-scratchpad: the sandbox pod image (anton + scratchpad boot + cloud_turn entrypoint).
# Consumed by scratchpad-controller (SCRATCHPAD_CONTROLLER__SCRATCHPAD_IMAGE) and Minds.
# The controller execs `python -m anton.cloud_turn` (whole turn) or
# `/usr/local/bin/scratchpad-boot.sh` (single cell) inside a gVisor pod running as UID 1000.
FROM python:3.12-slim

# uv is required at RUNTIME: scratchpad_boot shells out to `uv pip install` for missing packages
# via ANTON_UV_PATH=/usr/local/bin/uv, so uv must be present at exactly that path.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv

# hatch-vcs would otherwise need the .git history (excluded from the build context); the image
# only needs a valid version string, so pin one for the build.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=2.0.0 \
    VIRTUAL_ENV=/opt/anton-venv \
    PATH=/opt/anton-venv/bin:/usr/local/bin:$PATH \
    UV_LINK_MODE=copy

WORKDIR /app
COPY . /app

# Install anton into a venv owned by UID 1000 so the non-root runtime can also `uv pip install`
# missing packages at turn time. boto3 is imported by anton at runtime but not declared as a
# dependency, so add it explicitly.
RUN uv venv "$VIRTUAL_ENV" \
    && uv pip install --no-cache . boto3 \
    && chown -R 1000:1000 "$VIRTUAL_ENV"

# scratchpad-boot.sh: the single-cell entrypoint the controller execs (reads code + delimiter on stdin).
RUN printf '#!/bin/sh\nexec python -m anton.core.backends.scratchpad_boot\n' \
      > /usr/local/bin/scratchpad-boot.sh \
    && chmod 0755 /usr/local/bin/scratchpad-boot.sh

# Non-root, matching the pod securityContext (drop ALL caps, no service-account token, fs_group 1000, gVisor).
RUN useradd -u 1000 -m -s /bin/sh scratchpad
USER 1000

# The controller always execs an explicit command; this default keeps the image runnable standalone.
CMD ["python", "-m", "anton.cloud_turn"]
