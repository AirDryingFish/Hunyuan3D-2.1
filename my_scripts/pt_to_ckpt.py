import torch
import os

# 1. 载入原始 checkpoint
src_path = "/mnt/data/yangzengzhi/code/Hunyuan3D-2.1/hy3dshape/output_folder/dit/overfitting/ckpt/ckpt-step=00084000.ckpt/checkpoint/mp_rank_00_model_states.pt"
ckpt = torch.load(src_path, map_location="cpu")

# 如果深度学习框架把所有权重包在了 "module"、"model" 等字段里，可以先取出来
if "module" in ckpt:
    state = ckpt["module"]
elif "model" in ckpt and not any(k.startswith("model.") for k in ckpt):
    # 有些保存格式直接把 state_dict 放在 "model" 下
    state = ckpt["model"]
else:
    state = ckpt

# 2. 定义前缀
prefix_model       = "_forward_module.model."
prefix_vae         = "_forward_module.first_stage_model"
prefix_conditioner = "_forward_module.cond_stage_model"  # 根据实际再精细化

# 3. 拆分并去掉前缀
model_sd = {}
vae_sd   = {}
cond_sd  = {}

for k, v in state.items():
    if k.startswith(prefix_model):
        new_k = k[len(prefix_model):]
        model_sd[new_k] = v.half()  # 转 fp16

    # elif k.startswith(prefix_vae):
    #     new_k = k[len(prefix_vae):]
    #     vae_sd[new_k]   = v.half()
    # elif k.startswith(prefix_conditioner):
    #     # 如果 conditioner 的层级更深，可以类似上面再判断更多子前缀
    #     new_k = k[len(prefix_conditioner) + 1:]  # +1 去掉后面的点
    #     cond_sd[new_k]  = v.half()

import sys
root = os.path.abspath(os.path.join(__file__, "..", "..", "hy3dshape"))
sys.path.insert(0, root)

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from peft import get_peft_model, LoraConfig, PeftModel
base_id = "tencent/Hunyuan3D-2.1"
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')  # CPU 上就行，少占显存

lora_sd = model_sd
loraconfig = LoraConfig(
    r=16,
    target_modules=["to_q","to_k","to_v","out_proj","proj","net.2"]
)
hf_backbone = pipeline.model
peft_model = get_peft_model(hf_backbone, loraconfig)
peft_model.load_state_dict(lora_sd, strict=True)

# --- 4) merge & unload LoRA adapter ---
merged = peft_model.merge_and_unload()
# 4. 组装成最终 dict
out = {
    "model":       merged.state_dict(),
    "vae":         pipeline.vae.state_dict(),
    "conditioner": pipeline.conditioner.state_dict(),
}

# 5. 保存到目标路径
out_path = "/mnt/data/yangzengzhi/ckpts/h3d_merged/hunyuan3d-dit-v2-1/model.fp16.ckpt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(out, out_path)

print(f"Converted checkpoint saved to {out_path}")