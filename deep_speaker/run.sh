mkdir log

CUDA_VISIBLE_DEVICES=0 nohup python train.py > log/train.log &
