#!/bin/bash

T=`date +%m%d%H%M`
ROOT=../../../ # changed
#cfg=d4-rfcn-1x.yaml
cfg=apollo_config.yaml
#cfg=mono3d_config.yaml
#cfg=kitti_config.yaml
#cfg=retina-v11_apollo.yaml
#cfg=kitti_config_conv.yaml
#cfg=mono3d_config_conv.yaml
#cfg=demo_video_config.yaml
#export PYTHONPATH=$ROOT:$PYTHONPATH

# g=$(($2<8?$2:8))
# srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
#     --job-name=$3 \
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
#python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
#python -m pod test \
#  --config=$cfg \
#  2>&1 | tee log.test.$T
