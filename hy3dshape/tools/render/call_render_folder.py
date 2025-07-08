import os
import subprocess

SRC_DIR = "/mnt/data/yangzengzhi/data/shoes/"
DST_ROOT = "/mnt/data/yangzengzhi/data/shoe_train_data"

# 获取DST_ROOT下所有一级文件夹名
dst_folders = set(
    name for name in os.listdir(DST_ROOT)
    if os.path.isdir(os.path.join(DST_ROOT, name))
)

for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        if not fname.endswith(".obj"):
            continue
        obj_path = os.path.join(root, fname)
        base = os.path.splitext(fname)[0]
        # 只在DST_ROOT有同名一级文件夹时才渲染
        if base not in dst_folders:
            print(f"Skip {fname}: {base} not in DST_ROOT folders")
            continue
        out_dir = os.path.join(DST_ROOT, base, "render_cond")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "render.py", "--",
            "--object", obj_path,
            "--output_folder", out_dir,
            "--views", "24",
            "--resolution", "518",
            "--geo_mode"
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)