import os
import glob
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# Optional fix for torchvision compatibility
try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


def batch_paint(mesh_dir, image_dir, output_dir, max_num_view=6, resolution=512):
    """
    Process all .glb meshes in mesh_dir, find corresponding .jpg/.png images in image_dir,
    run the Hunyuan3DPaintPipeline, and save both .obj and .glb to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the painting pipeline
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    mesh_paths = glob.glob(os.path.join(mesh_dir, '*.glb'))
    if not mesh_paths:
        print(f"No .glb files found in {mesh_dir}")
        return

    for i, mesh_path in enumerate(mesh_paths, 1):
        print(f"Processing [{i}/{len(mesh_paths)}]: {mesh_path}")
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        # lookup image
        jpg = os.path.join(image_dir, base_name + '.jpg')
        png = os.path.join(image_dir, base_name + '.png')
        image_path = jpg if os.path.isfile(jpg) else (png if os.path.isfile(png) else None)
        if image_path is None:
            print(f"Warning: No matching image for {base_name}")
            continue

        # define output paths in advance
        output_obj = os.path.join(output_dir, f"{base_name}.obj")

        # run pipeline and force save to our path
        paint_pipeline(
            mesh_path=mesh_path,
            image_path=image_path,
            output_mesh_path=output_obj,
            use_remesh=True,
            save_glb=True
        )

        print(f"Saved painted OBJ to: {output_obj}")
        # check for corresponding .glb
        output_glb = output_obj.replace('.obj', '.glb')
        if os.path.isfile(output_glb):
            print(f"Saved painted GLB to: {output_glb}")
        else:
            print(f"Expected GLB not found: {output_glb}")


if __name__ == '__main__':
    # === User Configuration ===
    mesh_dir = '/mnt/data/yangzengzhi/data/test_splited/out_step_28000/'
    image_dir = '/mnt/data/yangzengzhi/data/test_splited/'
    output_dir = '/mnt/data/yangzengzhi/data/test_splited/out_textured_step_28000/'
    max_num_view = 6  # 6 to 9
    resolution = 512  # 512 or 768

    batch_paint(mesh_dir, image_dir, output_dir, max_num_view, resolution)
