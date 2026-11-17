#!/usr/bin/env bash
# Run once on the HPC LOGIN node (compute nodes may lack internet). Adjust module names after checking `module avail`.
set -euo pipefail
# module load anaconda3/personal   # Imperial RCS usually provides conda via a module/personal install -- verify
python3 -m venv "$HOME/venvs/fate" && source "$HOME/venvs/fate/bin/activate"
pip install -U pip && pip install -r requirements.txt && pip install -e .
# pre-download pretrained weights while we still have internet
python - <<'PY'
import torchvision
for a in ("resnet18", "resnet50"):
    getattr(torchvision.models, a)(weights="DEFAULT")
PY
echo "env ready"
