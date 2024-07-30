#!/bin/bash
export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib/

#CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc-per-node=2  mainttc.py
#CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc-per-node=2  train/convlstm/train.py
CUDA_VISIBLE_DEVICES=1,2,3,7 torchrun --nproc-per-node=4  --master-port 29660 train.py

