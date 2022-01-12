
# python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.2 --master_port 29502 train_detect_split.py --fold 0 --gpu 0
# python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.3 --master_port 29503 train_detect_split.py --fold 1 --gpu 6
# python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.4 --master_port 29504 train_detect_split.py --fold 2 --gpu 6
# python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.5 --master_port 29505 train_detect_split.py --fold 3 --gpu 7
python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.6 --master_port 29506 train_detect_split.py --fold 4 --gpu 5

# python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.4 --master_port 29504 train_detect_split.py --fold 2 --gpu 1