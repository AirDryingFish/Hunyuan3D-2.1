#!/usr/bin/env python3
"""
Batch 3D Shape Generation Script

Reads all image files in an input directory, removes their background if needed,
runs the Hunyuan3DDiTFlowMatchingPipeline to reconstruct a 3D mesh,
and exports each mesh to a .glb file in the output directory.

Usage:
    pip install pillow hy3dshape tqdm
    python batch_shapegen.py \
        --input_dir ./images \
        --output_dir ./meshes \
        --model_path tencent/Hunyuan3D-2.1

"""
import os
import argparse
from PIL import Image
from tqdm import tqdm

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

def process_image(image_path, pipeline, remover):
    img = Image.open(image_path).convert("RGBA")
    # If the image has no alpha channel, remove background
    if img.mode == 'RGB' or (img.mode == 'RGBA' and img.getchannel('A').getextrema()[1] == 255):
        img = remover(img)
    # Generate mesh
    mesh = pipeline(image=img, octree_resolution=1024, num_chunks=500000)[0]
    return mesh

def main():
    parser = argparse.ArgumentParser(description="Batch 3D shape generation from images")
    parser.add_argument("--input_dir", "-i", required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save output .glb meshes")
    parser.add_argument("--model_path", "-m", default="tencent/Hunyuan3D-2.1",
                        help="Pretrained model path or identifier")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize pipeline and background remover
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)
    # pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file('/mnt/data/yangzengzhi/ckpts/h3d_merged/hunyuan3d-dit-v2-1/model.fp16.ckpt', '/mnt/data/yangzengzhi/ckpts/h3d_merged/hunyuan3d-dit-v2-1/config.yaml')
    remover = BackgroundRemover()

    # Gather image files
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    image_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if os.path.splitext(f.lower())[1] in exts
    ]

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{base}.glb")
            
            if os.path.exists(out_path):
                print(f"skipping {out_path}")
                continue
            mesh = process_image(img_path, pipeline, remover)
            mesh.export(out_path)
        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {e}")

    print("Batch processing complete.")

if __name__ == "__main__":
    main()

