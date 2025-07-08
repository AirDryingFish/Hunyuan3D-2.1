# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import igl
import numpy as np
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from scipy.stats import truncnorm
import trimesh
import json

def random_sample_pointcloud(mesh, num = 30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]

def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:,i]], np.sum(VN[F[:,i]] * N, axis=-1)), axis=-1)
        VN2[F[:,i]] = np.min(dot, axis=-1)

    sharp_mask = VN2<0.985
    # collect edge
    edge_a = np.concatenate((F[:,0],F[:,1],F[:,2]))
    edge_b = np.concatenate((F[:,1],F[:,2],F[:,0]))
    sharp_edge = ((sharp_mask[edge_a] * sharp_mask[edge_b]))
    edge_a = edge_a[sharp_edge>0]
    edge_b = edge_b[sharp_edge>0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num,1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals

def sample_sdf(mesh, random_surface, sharp_surface):
    n_volume_points = sharp_surface.shape[0] * 2
    vol_points = (np.random.rand(n_volume_points, 3) - 0.5) * 2 * 1.05

    a, b = -0.25, 0.25
    mu = 0

    # get near points (add offset on surface points)
    offset1 = truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(len(random_surface), 3))
    offset2 = truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05,  size=(len(random_surface), 3))
    random_near_points = np.concatenate([
        random_surface + offset1,
        random_surface + offset2
    ], axis=0)

    unit_num = len(sharp_surface) // 6
    sharp_near_points = np.concatenate([
        sharp_surface[:unit_num] + np.random.normal(scale=0.001, size=(unit_num, 3)),
        sharp_surface[unit_num:unit_num*2] + np.random.normal(scale=0.003, size=(unit_num,3)),
        sharp_surface[unit_num*2:unit_num*3] + np.random.normal(scale=0.06, size=(unit_num,3)),
        sharp_surface[unit_num*3:unit_num*4] + np.random.normal(scale=0.01, size=(unit_num,3)),
        sharp_surface[unit_num*4:unit_num*5] + np.random.normal(scale=0.02, size=(unit_num,3)),
        sharp_surface[unit_num*5:] + np.random.normal(scale=0.04, size=(len(sharp_surface)-5*unit_num,3))
    ], axis=0)

    np.random.shuffle(random_near_points)
    np.random.shuffle(sharp_near_points)

    sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    try:
        vol_sdf, I, C, *rest = igl.signed_distance(
            vol_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False,
            sign_type=sign_type
            )
    except:
        vol_sdf, I, C, *rest = igl.signed_distance(
            vol_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False
            )
    try:
        random_near_sdf, I, C, *rest = igl.signed_distance(
            random_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False,
            sign_type=sign_type
            )
    except:
        random_near_sdf, I, C, *rest = igl.signed_distance(
            random_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False
            )
    try:
        sharp_near_sdf, I, C, *rest = igl.signed_distance(
            sharp_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False,
            sign_type=sign_type
            )
    except:
        sharp_near_sdf, I, C, *rest = igl.signed_distance(
            sharp_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            # return_normals=False
            )
        
    vol_label = -vol_sdf
    random_near_label = -random_near_sdf
    sharp_near_label = -sharp_near_sdf

    data = {
        "vol_points": vol_points.astype(np.float16),
        "vol_label": vol_label.astype(np.float16),
        "random_near_points": random_near_points.astype(np.float16),
        "random_near_label": random_near_label.astype(np.float16),
        "sharp_near_points": sharp_near_points.astype(np.float16),
        "sharp_near_label": sharp_near_label.astype(np.float16)
    }
    return data

def SampleMesh(V, F):
    print("sampling")
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=V, faces=F)

    area = mesh.area
    sample_num = 499712//4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    #save_surface
    surface = np.concatenate((random_surface, random_normal), axis = 1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)

    surface_data = {
        "random_surface": surface,
        "sharp_surface": sharp_surface,
    }
    # print(f"sampling sdf")
    # sdf_data = sample_sdf(mesh, random_surface, random_sharp_surface)
    # return surface_data, sdf_data
    return surface_data

def normalize_to_unit_box(V):
    """
    Normalize the vertices V to fit inside a unit bounding box [0,1]^3.
    V: (n,3) numpy array of vertex positions.
    Returns: normalized V
    """
    # V_min = V.min(axis=0)
    # V_max = V.max(axis=0)
    # scale = (V_max - V_min).max() * 1.01
    # V_normalized = (V - V_min) / scale
    # return V_normalized
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    center = (V_min + V_max) / 2
    scale = (V_max - V_min).max() * 1.01 / 2
    V_norm = (V - center) / scale
    return V_norm

# Given: V (n x 3 array of vertices), F (m x 3 array of faces)
# Parameters epsilon/grid_res
def Watertight(V, F, epsilon = 2.0/512, grid_res = 512):
    # Compute bounding box
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # Create a uniform grid
    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute SDF at grid points using igl.signed_distance with pseudo normals
    sdf, *rest = igl.signed_distance(
        grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
 
    # igl.marching_cubes returns (vertices, faces)
    mc_verts, mc_faces, *rest = igl.marching_cubes(epsilon - np.abs(sdf), grid_points, grid_res, grid_res, grid_res, 0.0)

    # mc_verts: (k x 3) array of vertices of the epsilon contour
    # mc_faces: (l x 3) array of faces of the epsilon contour
    return mc_verts, mc_faces

CAMERA_DIST = 3.0
NUM_VIEWS = 50
IMG_SIZE = 512

def id_to_rgb01(idx: int):
    # idx 从 1 开始，保证 0 作为背景
    r = (idx >> 16) & 0xFF
    g = (idx >> 8 ) & 0xFF
    b = (idx      ) & 0xFF
    return np.array([r, g, b], dtype=np.float32)/255.0

# helper: 将 uint8 RGB 解码回整数 ID
def rgb8_to_id(rgb8: np.ndarray):
    # rgb8: H×W×3 uint8
    flat = rgb8.reshape(-1, 3).astype(np.uint32)
    return (flat[:,0] << 16) | (flat[:,1] << 8) | flat[:,2]


def _generate_camera_poses(num_views=NUM_VIEWS, dist=CAMERA_DIST):
    """
    Create world-to-camera poses on a sphere around the origin.
    Returns list of 4x4 camera transforms for pyrender (camera looks along -Z axis).
    """
    poses = []
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        phi = np.pi / 6
        x = dist * np.cos(phi) * np.cos(theta)
        y = dist * np.cos(phi) * np.sin(theta)
        z = dist * np.sin(phi)
        eye = np.array([x, y, z], dtype=np.float32)
        target = np.zeros(3, dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        # compute camera basis: right, true up, forward (view direction)
        # pyrender expects camera -Z axis to point towards the scene
        forward = eye - target
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        # assemble rotation: columns are basis vectors
        # local X->right, local Y->true_up, local Z->forward
        R = np.stack([right, true_up, forward], axis=1)
        # pose maps camera to world
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = eye
        poses.append(pose)
    return poses

def process_obj(input_obj: str, output_prefix: str):
    # 读取网格
    V, F = igl.read_triangle_mesh(input_obj)
    # 如果没有顶点，则跳过
    if V is None or V.shape[0] == 0:
        print(f"警告: 在文件 {input_obj} 中未读到顶点，跳过处理")
        return

    # 归一化到单位盒
    V = normalize_to_unit_box(V)

    # 保证网格闭合并采样生成 surface 与 sdf 数据
    mc_verts, mc_faces = Watertight(V, F)

    # ——— 只保留最大的连通组件 ———
    full_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces)

    components = full_mesh.split(only_watertight=False)
    N_comp = len(components)
    print(f"Found {N_comp} components")
    import pyrender

    # with pyrender.OffscreenRenderer(
    #     viewport_width=IMG_SIZE,
    #     viewport_height=IMG_SIZE,
    # ) as renderer:

    # create vis directory
    # vis_dir = os.path.splitext(remesh_path)[0] + '_vis_colorid'
    # os.makedirs(vis_dir, exist_ok=True)

    # build scene with ID materials
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[1,1,1])
    for idx, comp in enumerate(components):
        vid = idx+1
        rgb01 = id_to_rgb01(vid)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*rgb01,1.0], metallicFactor=0.0, roughnessFactor=1.0)
        node = pyrender.Mesh.from_trimesh(comp, material=mat, smooth=False)
        scene.add(node)

    # render each view and save color_id image
    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    cam_poses = _generate_camera_poses()
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    visible_ids = set()

    for i, pose in enumerate(cam_poses):
        cam_node = scene.add(camera, pose=pose)
        color_buf, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA| pyrender.RenderFlags.FLAT)
        scene.remove_node(cam_node)
        # save pure ID render
        # filename = os.path.join(vis_dir, f'view_{i:02d}_id.png')
        # imageio.imwrite(filename, color_buf)
        # decode visible IDs
        ids_all = rgb8_to_id(color_buf[...,:3].astype(np.uint8))
        unique = np.unique(ids_all)
        visible_ids.update(unique[unique>0])

    renderer.delete()

    # filter and export
    vis_idxs = [vid-1 for vid in visible_ids]
    if not vis_idxs:
        print("No visible components, exporting full mesh.")
        final = full_mesh
    else:
        print(f"{len(vis_idxs)}/{N_comp} visible components")
        final = trimesh.util.concatenate([components[i] for i in vis_idxs])
    try:
        # print(f"faces = {final.faces.shape[0]}")
        # target_faces = 2000_000
        # if final.faces.shape[0] > target_faces:
        #     final = final.simplify_quadratic_decimation(target_faces)
        #     print(f"简化到 {len(final.faces)} faces 以降低后续计算量")
        mc_verts, mc_faces = final.vertices, final.faces

        # surface_data, sdf_data = SampleMesh(mc_verts, mc_faces)
        surface_data = SampleMesh(mc_verts, mc_faces)

        # 创建输出目录
        parent_folder = os.path.dirname(output_prefix)
        os.makedirs(parent_folder, exist_ok=True)

        # 保存结果
        np.savez(f'{output_prefix}_surface.npz', **surface_data)
        # np.savez(f'{output_prefix}_sdf.npz', **sdf_data)
        igl.writeOBJ(f'{output_prefix}_watertight.obj', mc_verts, mc_faces)
    except Exception as e:
        print("SampleMesh 运行时报错：", e)
        return


from concurrent.futures import ProcessPoolExecutor, as_completed

def worker(entry, output_dir):
    # 根据 entry 决定 obj_path 和 base_name
    if isinstance(entry, str):
        obj_path = entry
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
    else:
        obj_path = entry['path']
        base_name = entry.get('name', os.path.splitext(os.path.basename(obj_path))[0])

    # 设定输出前缀
    if output_dir:
        output_folder = os.path.join(output_dir, base_name)
    else:
        parent = os.path.dirname(obj_path)
        output_folder = os.path.join(parent, base_name)

    print(f"Processing {obj_path} -> {output_folder}")
    # 调用原来的处理函数
    process_obj(obj_path, output_folder)
    return obj_path
from concurrent.futures import ProcessPoolExecutor, as_completed
def main():
    parser = argparse.ArgumentParser(
        description='Batch process OBJ files listed in a JSON and output surface/SDF data (multi-process).'
    )
    parser.add_argument(
        '--json_path', type=str,
        default='/mnt/data/yangzengzhi/data/objs.json',
        help='Path to JSON file listing OBJ entries'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/mnt/data/yangzengzhi/data/shoe_processed_512/',
        help='Directory to save outputs (default: same folder as each OBJ)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel worker processes (default: CPU count)'
    )
    args = parser.parse_args()

    # 加载 JSON
    with open(args.json_path, 'r') as f:
        obj_entries = json.load(f)

    # 建议把 entries 转成 list，确保长度可知
    entries = list(obj_entries)

    # 并行执行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = {executor.submit(worker, entry, args.output_dir): entry for entry in entries}

        # 可选：监控进度
        for future in as_completed(futures):
            entry = futures[future]
            try:
                obj_path = future.result()
                print(f"[Done] {obj_path}")
            except Exception as e:
                print(f"[Error] {entry}: {e}")

if __name__ == '__main__':
    main()