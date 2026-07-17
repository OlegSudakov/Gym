#!/bin/bash
source .env
python optimize.py \
    --agent-url http://127.0.0.1:14239 \
    --train-data "data/train.jsonl" \
    --val-data "data/validation.jsonl" \
    --test-data "data/test.jsonl" \
    --max-calls 1000 \
    --reflection-model ${REFLECTION_MODEL} \
    --reflection-base-url ${REFLECTION_BASE_URL} \
    --reflection-api-key ${REFLECTION_API_KEY}