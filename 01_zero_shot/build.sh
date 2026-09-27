#!/bin/bash
# Builds 01_zero_shot_submission.zip, ready to upload to Codabench.
set -euo pipefail
cd "$(dirname "$0")"
rm -f ../01_zero_shot_submission.zip
zip -q -r ../01_zero_shot_submission.zip main.py
echo "-> ../01_zero_shot_submission.zip"
