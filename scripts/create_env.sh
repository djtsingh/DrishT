#!/usr/bin/env bash
# Bash script for WSL/macOS to create venv and install light requirements
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "To install heavy frameworks (TensorFlow/PyTorch), uncomment the lines in requirements.txt and run again."