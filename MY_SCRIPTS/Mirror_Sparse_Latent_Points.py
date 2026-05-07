import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--npz_path", required=True, help="Input NPZ file")
parser.add_argument("--xyz_dir", required=True, help="Directory with XYZ visualization files")
parser.add_argument("--output_npz", required=True, help="Output mirrored NPZ file")
parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for validation")
parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for validation")
parser.add_argument("--strict", action="store_true", help="Stop if validation with XYZ fails")

args = parser.parse_args()

npz_path = args.npz_path
xyz_dir = args.xyz_dir
output_npz = args.output_npz
atol = args.atol
rtol = args.rtol

# Rotación temporal: lleva la nube a la orientación deseada
R_left = np.array([
    [ 0.0, 0.0,  1.0],
    [ 0.0, 1.0,  0.0],
    [-1.0, 0.0,  0.0]
])

# Para puntos como filas: rotated = points @ R_left.T
# La inversa se aplica como: restored = rotated @ R_left
R_inverse_for_rows = R_left


def validate_npz_structure(npz_data):
    print("\n[1] Contenido del NPZ:")
    for k in npz_data.files:
        print(f"  {k}: shape = {npz_data[k].shape}, dtype = {npz_data[k].dtype}")

    if "points" not in npz_data.files:
        raise KeyError("El NPZ no contiene la clave 'points'.")

    points = npz_data["points"]

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Se esperaba points con shape (B, N, 3), pero se obtuvo {points.shape}")

    print("\n[OK] Estructura de points válida.")
    print(f"  points shape: {points.shape}")
    print(f"  points dtype: {points.dtype}")

    return points


def validate_npz_against_xyz(points, xyz_dir, atol=1e-6, rtol=1e-6, strict=False):
    print("\n[2] Validando NPZ contra archivos XYZ...")

    xyz_files = sorted([f for f in os.listdir(xyz_dir) if f.endswith(".xyz")])

    print(f"  Nubes en NPZ: {points.shape[0]}")
    print(f"  Archivos XYZ encontrados: {len(xyz_files)}")

    if len(xyz_files) != points.shape[0]:
        print("[WARNING] El número de archivos XYZ no coincide con el número de nubes en el NPZ.")

    num_to_check = min(len(xyz_files), points.shape[0])

    mismatch_count = 0
    max_global_diff = 0.0

    for i in range(num_to_check):
        xyz_path = os.path.join(xyz_dir, xyz_files[i])
        xyz_data = np.loadtxt(xyz_path)

        if xyz_data.ndim == 1:
            xyz_data = xyz_data[None, :]

        xyz_points = xyz_data[:, :3]
        npz_points = points[i]

        if xyz_points.shape != npz_points.shape:
            print(f"[MISMATCH SHAPE] index={i}, file={xyz_files[i]}")
            print(f"  XYZ shape: {xyz_points.shape}")
            print(f"  NPZ shape: {npz_points.shape}")
            mismatch_count += 1
            continue

        diff = np.abs(xyz_points - npz_points)
        max_diff = diff.max()
        max_global_diff = max(max_global_diff, max_diff)

        if not np.allclose(xyz_points, npz_points, atol=atol, rtol=rtol):
            print(f"[MISMATCH VALUES] index={i}, file={xyz_files[i]}, max_diff={max_diff}")
            mismatch_count += 1

    print("\nResultado validación NPZ vs XYZ:")
    print(f"  Comparados: {num_to_check}")
    print(f"  Mismatches: {mismatch_count}")
    print(f"  Max global diff: {max_global_diff}")

    if mismatch_count == 0:
        print("[OK] Los puntos del NPZ coinciden con los XYZ.")
    else:
        print("[WARNING] Hay diferencias entre el NPZ y los XYZ.")
        if strict:
            raise ValueError("La validación contra XYZ falló y --strict está activado.")


def mirror_points(points):
    """
    Aplica:
    1. Rotar temporalmente
    2. Mirror en X en la orientación temporal
    3. Devolver a orientación original
    """

    # 1. Rotar
    rotated_points = points @ R_left.T

    # 2. Mirror en X dentro del sistema rotado
    mirrored_rotated_points = rotated_points.copy()
    mirrored_rotated_points[..., 0] *= -1

    # 3. Desrotar
    mirrored_points = mirrored_rotated_points @ R_inverse_for_rows

    return mirrored_points


def validate_output_npz(original_npz, output_npz_path):
    print("\n[4] Validando NPZ reflejado...")

    mirrored_npz = np.load(output_npz_path, allow_pickle=True)

    original_keys = set(original_npz.files)
    mirrored_keys = set(mirrored_npz.files)

    if original_keys != mirrored_keys:
        raise ValueError(f"Las claves no coinciden. Original={original_keys}, Mirrored={mirrored_keys}")

    for k in original_npz.files:
        if original_npz[k].shape != mirrored_npz[k].shape:
            raise ValueError(f"Shape distinto en clave {k}: original={original_npz[k].shape}, mirrored={mirrored_npz[k].shape}")

        if original_npz[k].dtype != mirrored_npz[k].dtype:
            raise ValueError(f"Dtype distinto en clave {k}: original={original_npz[k].dtype}, mirrored={mirrored_npz[k].dtype}")

    print("[OK] El NPZ reflejado conserva las mismas claves, shapes y dtypes.")

    print("\nContenido del NPZ reflejado:")
    for k in mirrored_npz.files:
        print(f"  {k}: shape = {mirrored_npz[k].shape}, dtype = {mirrored_npz[k].dtype}")


# ==========================
# 1. Cargar NPZ original
# ==========================

npz_data = np.load(npz_path, allow_pickle=True)
points = validate_npz_structure(npz_data)

# ==========================
# 2. Evaluar si NPZ coincide con XYZ
# ==========================

validate_npz_against_xyz(
    points=points,
    xyz_dir=xyz_dir,
    atol=atol,
    rtol=rtol,
    strict=args.strict
)

# ==========================
# 3. Aplicar mirror
# ==========================

print("\n[3] Aplicando mirror...")

mirrored_points = mirror_points(points)
mirrored_points = mirrored_points.astype(points.dtype)

print("  Original points shape:", points.shape)
print("  Mirrored points shape:", mirrored_points.shape)
print("  Original dtype:", points.dtype)
print("  Mirrored dtype:", mirrored_points.dtype)

# ==========================
# 4. Guardar NPZ reflejado
# ==========================

output_dict = {}

for k in npz_data.files:
    if k == "points":
        output_dict[k] = mirrored_points
    else:
        output_dict[k] = npz_data[k]

output_dir = os.path.dirname(output_npz)
if output_dir != "":
    os.makedirs(output_dir, exist_ok=True)

np.savez(output_npz, **output_dict)

print("\nNPZ reflejado guardado en:")
print(output_npz)

# ==========================
# 5. Evaluar NPZ reflejado
# ==========================

validate_output_npz(npz_data, output_npz)

print("\nProceso terminado correctamente.")

"""
Call as: python MY_SCRIPTS/Mirror_Sparse_Latent_Points.py --npz_path  --xyz_dir  --output_npz 

python Mirror_Sparse_Latent_Points.py --npz_path /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/shapenet_psr_generated_data_16_pts.npz --xyz_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/shapenet_psr_generated_data_16_pts_visualization --output_npz /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz
"""