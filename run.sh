# torchrun --nproc_per_node=1 --nnodes=1 train.py --config config/celeba.yaml --use_amp
torchrun --nproc_per_node=1 --nnodes=1 train.py --config config/celeba.yaml --use_amp --latent_traverse
