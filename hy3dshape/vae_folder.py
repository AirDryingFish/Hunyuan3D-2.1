import os
from pathlib import Path
import torch
from tqdm import tqdm

from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh

# Initialize VAE model
vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    use_safetensors=False,
    variant='fp16',
)

# Move to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = vae.to(device)
vae.eval()

# Initialize surface loader
loader = SharpEdgeSurfaceLoader(
    num_sharp_points=0,
    num_uniform_points=81920,
)

# Input and output directories
input_dir = Path('/mnt/data/yangzengzhi/data/shoe_processed/')
output_dir = Path('/mnt/data/yangzengzhi/data/shoe_processed_vae/')
output_dir.mkdir(exist_ok=True)

# Batch process all .obj files
for obj_path in tqdm(list(input_dir.glob('*.obj')), desc="Processing meshes"):
    try:
        # Load surface points
        surface = loader(str(obj_path)).to(device, dtype=torch.float16)

        # Encode and decode to get latents
        latents = vae.encode(surface)
        latents = vae.decode(latents)

        # Reconstruct mesh from latents
        mesh_data = vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=1.01,
            mc_level=0.0,
            num_chunks=20000,
            octree_resolution=256,
            mc_algo='mc',
            enable_pbar=False
        )

        # Convert and export
        mesh = export_to_trimesh(mesh_data)[0]
        save_path = output_dir / obj_path.name
        mesh.export(str(save_path))
    except Exception as e:
        print(f"Failed to process {obj_path.name}: {e}")

print("Batch processing complete. Output saved to:", output_dir)
