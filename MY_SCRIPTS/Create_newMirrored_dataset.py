import numpy as np
import open3d as o3d
import os
import argparse

def householder_transformation(pc):
    try:
        # Define the vector "v" for the YZ plane - Transformation in X
        v = np.array([1,0,0])
        
        # Create the Householder matrix
        H = np.eye(3) - 2 * np.outer(v,v) # Entity matrix

        # Apply transformation
        transformed_points = pc @ H.T # (N, 3) x (3, 3) - Matrix product

        return transformed_points # Return the transformed points
    except:
        raise Exception("Error while trying to do the householder transformation.")

def farthest_point_sampling(points, n_samples):
    # Create a point cloud of Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # compute FPS
    downpcd_farthest = pcd.farthest_point_down_sample(n_samples)

    return np.asarray(downpcd_farthest.points)

def compute_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    return normals

# To rotate the PC to the left
R_left = np.array([
    [ 0.0, 0.0,  1.0],
    [ 0.0, 1.0,  0.0],
    [-1.0, 0.0,  0.0]
])

# Argument parser
parser = argparse.ArgumentParser(description="Create Dataset for SLIDE3D")
parser.add_argument("--input_path", type=str, help="Input path")
parser.add_argument("--output_path", type=str, help="Output path")
parser.add_argument("--device", type=str, help="cuda")

args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

input_path = args.input_path
output_path = args.output_path
device = args.device
num_points = 15000

processed = 0
skipped = 0

for model_id in sorted(os.listdir(input_path)):
    model_input_dir = os.path.join(input_path, model_id)
    if not os.path.isdir(model_input_dir):
        skipped += 1
        continue

    input_npz = os.path.join(model_input_dir, "pointcloud.npz")
    if not os.path.isfile(input_npz):
        print(f"[WARN] No pointcloud.npz en {model_id}")
        skipped += 1
        continue

    print(f"Processing: {model_id}")

    # ---------------------
    # Load file
    # ---------------------
    data = np.load(input_npz)

    points = data['points']
    #normals = data['normals']

    points = points @ R_left.T

    # Clone the cloud and apply HH transformation
    cloned_points = points.copy()#.clone()
    cloned_points = householder_transformation(cloned_points)

    # Joint with original cloud
    full_pc = np.concatenate([points, cloned_points], axis = 0) # [2N, 3]

    # Apply FPS
    full_pc = farthest_point_sampling(full_pc, num_points)

    # Rotate it back
    R_right = R_left.T
    full_pc = full_pc @ R_right.T

    # Compute normals
    normals = compute_normals(full_pc)

    # Save the resulting cloud
    model_output_dir = os.path.join(output_path, model_id)
    os.makedirs(model_output_dir, exist_ok=True)

    output_npz = os.path.join(model_output_dir, "pointcloud.npz")
    np.savez(output_npz, points=full_pc, normals=normals)

    processed += 1

print("Procesamiento completo")
print(f"Procesados: {processed}")
print(f"Ignorados: {skipped}")