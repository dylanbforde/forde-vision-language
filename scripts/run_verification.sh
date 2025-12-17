#!/usr/bin/env bash
set -e

echo "--- Running Verification Scripts ---"

echo "1. Verifying Gradient Sink Pattern..."
python tests/reproduce_grad_sink.py

echo "2. Verifying Model Initialization..."
python tests/verify_init.py

echo "3. Verifying Hoyer Sparsity..."
python tests/verify_hoyer.py

echo "4. Verifying Stats Buffer Reset..."
python tests/verify_reset.py

echo "--- All Verification Scripts Passed ---"
