#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco

python scripts/dual_arm_pick_place.py "$@"
