#!/usr/bin/env bash
# 禁用 RDMA/IB，只用默认通道
export NCCL_IB_DISABLE=1
#（可选）如果你真的要强制 TCP，用 lo
# export NCCL_SOCKET_IFNAME=lo

export NCCL_DEBUG=WARN

export node_num=1
export node_rank=0
export master_ip=127.0.0.1
export config=configs/hunyuandit-finetuning-flowmatching-dinog518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/overfitting

node_num=$node_num
node_rank=$node_rank
master_ip=$master_ip
config=$config
output_dir=$output_dir

echo "node_num  = $node_num"
echo "node_rank = $node_rank"
echo "master_ip = $master_ip"
echo "config    = $config"
echo "output_dir= $output_dir"
mkdir -p "$output_dir"
cp "$config" "$output_dir"

python3 main.py \
  --num_nodes $node_num \
  --num_gpus 8 \
  --config $config \
  --output_dir $output_dir \
  --deepspeed