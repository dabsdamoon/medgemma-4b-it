#!/bin/bash
# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    echo "Create .env with RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"
    exit 1
fi

python tests/test_batch.py /Users/dabsdamoon/projects/houmy/data_acog_markdown --limit 1
