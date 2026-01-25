#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    echo "Create .env with RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate medgemma

# Run batch processing

python tests/test_batch.py raw --output summaries --limit 1


