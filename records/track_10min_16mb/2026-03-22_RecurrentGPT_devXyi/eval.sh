#!/bin/bash
# Run one seed. Usage: SEED=42 bash eval/eval.sh
# Runs from inside the records folder.

SEED=${SEED:-42}

DATA_PATH=${DATA_PATH:-../../../../data/datasets/fineweb10B_sp1024}
TOKENIZER_PATH=${TOKENIZER_PATH:-../../../../data/tokenizers/fineweb_1024_bpe.model}

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  SEED=$SEED \
  DATA_PATH=$DATA_PATH \
  TOKENIZER_PATH=$TOKENIZER_PATH \
  2>&1 | tee train_seed${SEED}.log
