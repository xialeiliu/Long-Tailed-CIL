#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"
RESULTS_DIR="$PROJECT_DIR/results"

if [ "$7" != "" ]; then
    RESULTS_DIR=$7
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for SEED in 0 
do
  if [ "$3" = "base" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_base_${SEED} \
                 --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed $SEED \
                 --nepochs 200 --batch-size 128 --results-path $RESULTS_DIR \
                 --approach $1 --gpu $2 --lr 0.1 --lr-min 1e-5 --lr-factor 3 --momentum 0.9 \
                 --weight-decay 0.0002 --lr-patience 15
    elif [ "$3" = "cifar100" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_fixd_${SEED} \
                --datasets "$3_$4" --num-tasks $6 --network resnet32 --seed $SEED \
                --nepochs 1 --batch-size 128 --results-path "$PROJECT_DIR/cifar100/$5base_$6tasks" \
                --approach $1 --gpu $2 --lr 0.1 --lr-factor 10 --momentum 0.9 \
                --weight-decay 5e-4 \
                --nc-first-task $5 \
                --num-exemplars-per-class 20 --exemplar-selection herding
    elif [ "$3" = "imagenet_subset" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_fixd_${SEED} \
                --datasets "$3_$4" --num-tasks $6 --network resnet18 --seed $SEED \
                --nepochs 90 --batch-size 128 --results-path "$PROJECT_DIR/imagenet_subset/$5base_$6tasks" \
                --approach $1 --gpu $2 --lr 0.1 --lr-factor 10 --momentum 0.9 \
                --weight-decay 1e-4 \
                --nc-first-task $5 --schedule_step 30 60 \
                --num-exemplars-per-class 20 --exemplar-selection herding
  else
          echo "No scenario provided."
  fi
done