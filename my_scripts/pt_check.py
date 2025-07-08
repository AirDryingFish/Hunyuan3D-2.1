import torch

# 1. 载入 checkpoint（根据模型大小选择 map_location）
# checkpoint = torch.load("/mnt/data/yangzengzhi/code/Hunyuan3D-2.1/hy3dshape/output_folder/dit/overfitting/ckpt/ckpt-step=00028000.ckpt/checkpoint/mp_rank_00_model_states.pt", map_location="cpu")
checkpoint = torch.load("/mnt/data/yangzengzhi/ckpts/model_28000_steps.fp16.ckpt", map_location="cpu")

# 2. 提取真正的 state_dict（不同框架存储格式可能略有差异）
if "module" in checkpoint:
    state_dict = checkpoint["module"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

# 3. 遍历并打印参数名与形状
for name, tensor in state_dict.items():
    print(f"{name:60s} {tuple(tensor.shape)}")