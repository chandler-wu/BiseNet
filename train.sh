
'''
NOTE: replace torchrun with torch.distributed.launch if you use older version of pytorch. I suggest you use the same version as I do since I have not tested compatibility with older version after updating.
'''

export CUDA_VISIBLE_DEVICES=2,3
cfg_file=/workspace/gitlab/qianwu/code/BiSeNet/configs/jingke/edge_busbar_piece_20230807.py
NGPUS=2
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file
