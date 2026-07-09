#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco

python manipulation_benchmark.py --viewer mujoco --keep-viewer-open --steps 400 --print-every 50 --policy pick "$@"
