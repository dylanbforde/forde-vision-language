#!/usr/bin/env bash
set -e

# Run verification scripts
/app/scripts/run_verification.sh

# Execute the command passed to the container
exec "$@"
