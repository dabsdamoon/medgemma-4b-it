#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate medgemma

# Run the summarization script
# Usage: ./run_summarization.sh [limit]
# Example: ./run_summarization.sh 5 (processes first 5 files)
# Default: Processes ALL files in the input directory

if [ -n "$1" ]; then
    LIMIT_ARG="--limit $1"
    echo "Starting batch summarization with limit: $1"
else
    LIMIT_ARG=""
    echo "Starting batch summarization for ALL files in raw/"
fi

python scripts/summarize_docs.py $LIMIT_ARG --input_dir raw --output_dir summaries
