# If: ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found 
# Do: ln /usr/local/gcc-8.3/lib64/libstdc++.so.6 -sf /usr/lib64/libstdc++.so.6

export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export node_num=1
export node_rank=0
export master_ip=10.25.25.83 # set your master_ip
export config=configs/hunyuandit-finetuning-flowmatching-dinog518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/overfitting
export NCCL_IB_DISABLE=1
NET_TYPE="high"
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond0
    export UCX_NET_DEVICES=bond0
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0
fi
export NCCL_DEBUG=WARN

node_num=$node_num
node_rank=$node_rank
master_ip=$master_ip
config=$config
output_dir=$output_dir

# config='configs/dit-from-scratch-overfitting-flowmatching-dinog518-bf16-lr1e4-1024.yaml'
# output_dir='output_folder/dit/overfitting_10'

echo node_num $node_num
echo node_rank $node_rank
echo master_ip $master_ip
echo config $config
echo output_dir $output_dir
echo NCCL_SOCKET_IFNAME $NCCL_SOCKET_IFNAME

if test -d "$output_dir"; then
    cp $config $output_dir
else
    mkdir -p "$output_dir"
    cp $config $output_dir
fi
export NCCL_IB_DISABLE=1

NODE_RANK=$node_rank \
HF_HUB_OFFLINE=0 \
MASTER_PORT=12345 \
MASTER_ADDR=$master_ip \
NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
NCCL_IB_GID_INDEX=3 \
NCCL_NVLS_ENABLE=0 \
python3 main.py \
    --num_nodes $node_num \
    --num_gpus 8 \
    --config $config \
    --output_dir $output_dir \
    --deepspeed
