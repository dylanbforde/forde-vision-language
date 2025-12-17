#!/usr/bin/env bash
set -e

echo "--- Running Verification Scripts ---"

echo "1. Verifying Gradient Sink Pattern..."
python reproduce_grad_sink.py

echo "2. Verifying Model Initialization..."
python verify_init.py

echo "3. Verifying Hoyer Sparsity..."
python verify_hoyer.py

echo "4. Verifying Stats Buffer Reset..."
python verify_reset.py

echo "--- All Verification Scripts Passed ---"
