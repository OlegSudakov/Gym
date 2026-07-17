"""Prepare train/validation/test splits for the instruction_following environment.

Downloads the nvidia/Nemotron-RL-instruction_following dataset (a single JSONL
already in NeMo Gym Responses-API format), randomly samples disjoint splits, and
writes them as .jsonl files whose rows match
resources_servers/instruction_following/data/example.jsonl.
"""

import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "nvidia/Nemotron-RL-instruction_following"
FILENAME = "instruction_following.jsonl"

TRAIN_N = 150
VAL_N = 200
TEST_N = 200
SEED = 42

ROW_KEYS = ["id", "instruction_id_list", "prompt", "kwargs", "responses_create_params"]

# Instructions whose checkers crash on this dataset's kwargs. count:count_increment_word
# expects string keyword1/keyword2 but they are stored as single-element lists, so its
# build_description() raises "'list' object has no attribute 'strip'". Drop such rows.
EXCLUDE_INSTRUCTION_IDS = {"count:count_increment_word"}

OUT_DIR = Path(__file__).parent


def load_rows():
    path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if not EXCLUDE_INSTRUCTION_IDS.intersection(row.get("instruction_id_list", []))]


def write_split(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in ROW_KEYS}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def prepare_dataset():
    rows = load_rows()
    if len(rows) < TRAIN_N + VAL_N + TEST_N:
        raise ValueError(f"Dataset has only {len(rows)} rows, need {TRAIN_N + VAL_N + TEST_N}.")

    rng = random.Random(SEED)
    sample = rng.sample(rows, TRAIN_N + VAL_N + TEST_N)
    train = sample[:TRAIN_N]
    val = sample[TRAIN_N : TRAIN_N + VAL_N]
    test = sample[TRAIN_N + VAL_N :]

    write_split(train, OUT_DIR / "train.jsonl")
    write_split(val, OUT_DIR / "validation.jsonl")
    write_split(test, OUT_DIR / "test.jsonl")


if __name__ == "__main__":
    prepare_dataset()
