#!/usr/bin/env bash
# Run on a machine WITH internet (your laptop / HPC login node), not in the restricted cloud container.
# URLs are from memory and NOT verified from this environment -- if one 404s, get the file from the dataset's official page.
set -euo pipefail
D="${FATE_DATA:-$(dirname "$0")/../data}"; mkdir -p "$D"; cd "$D"
if [ ! -d CUB_200_2011 ]; then
  curl -L -o CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
  tar xzf CUB_200_2011.tgz && rm CUB_200_2011.tgz
fi
echo "now: python scripts/check_data.py"
