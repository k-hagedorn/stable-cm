This is a custom implementation of https://arxiv.org/pdf/2410.11081
To install refer to zigma repository  

The Attention imported in models has to be modiefied as to not used a fused kernel for JVP to work


torchrun --nnodes=1 --nproc_per_node=1 train.py  --data-path /mnt/g/Dataset/datasets/train_10k --global-batch-size 1




![Sample from sCM training from scratch at step 30000](samples_30000_91d9ab6387341ae16927.png)

![Sample from sCM training from scratch at step 205000](samples_205000_d5b640a265019faa4534.png)
