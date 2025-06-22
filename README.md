This is a custom implementation of https://arxiv.org/pdf/2410.11081
Sampling function is still missing 

The Attention imported in models has to be modiefied as to not used a fused kernel for JVP to work


torchrun --nnodes=1 --nproc_per_node=1 train.py  --data-path /mnt/g/Dataset/datasets/train_10k --global-batch-size 1
