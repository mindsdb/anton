# Builder image with tools for building pip packages
FROM python:3.13.2 AS builder

# Install system dependencies required for pymssql
RUN apt-get update && apt-get install -y \
    freetds-dev \
    && apt-get clean

COPY requirements/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


# Final image
FROM python:3.13.2-slim AS final

WORKDIR /minds

# "rm ... docker-clean" stops docker from removing packages from our cache
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#example-cache-apt-packages
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=target=/var/lib/apt,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    freetds-dev \
    unixodbc-dev \
    curl libexpat1 \
    && apt-get clean

COPY . .
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONPATH "."
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 9010

CMD ["python", "-m", "uvicorn", "minds.server:app", "--host", "0.0.0.0", "--port", "9010"]
