#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-hyperagents-cpu}"

cd "${ROOT_DIR}"
docker build --network=host -f Dockerfile.cpu -t "${IMAGE_NAME}" .
