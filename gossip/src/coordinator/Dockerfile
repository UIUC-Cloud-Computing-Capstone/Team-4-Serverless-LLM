# Lightweight Dockerfile to run the central coordinator latency runner
# NOTE: This image copies the repository and installs Python requirements. Depending
# on the full project's dependencies this can be heavy; adapt as needed.

FROM python:3.11-slim

WORKDIR /app

# Install system deps that might be needed (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . /app

# Install Python requirements if present; ignore failure to allow quick iteration
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt || true; fi

# Ensure the gossip src is on PYTHONPATH
ENV PYTHONPATH=/app/gossip/src

ENTRYPOINT ["python", "gossip/src/coordinator/main.py"]
