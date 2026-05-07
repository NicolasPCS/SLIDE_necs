import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--npz_path", required=True, help="Path del NPZ reflejado")
parser.add_argument("--output_dir", required=True, help="Carpeta donde guardar las nubes XYZ")
parser.add_argument("--output_npz", default="", help="Path opcional para guardar un NPZ con las primeras nubes")
parser.add_argument("--num_clouds", type=int, default=5, help="Número de nubes a extraer")

args = parser.parse_args()

npz_path = args.npz_path
output_dir = args.output_dir
output_npz = args.output_npz
num_clouds = args.num_clouds

os.makedirs(output_dir, exist_ok=True)

# ==========================
# Cargar NPZ
# ==========================

data = np.load(npz_path, allow_pickle=True)

print("Contenido del NPZ:")
for k in data.files:
    print(f"  {k}: shape = {data[k].shape}, dtype = {data[k].dtype}")

if "points" not in data.files:
    raise KeyError("El NPZ no contiene la clave 'points'.")

points = data["points"]

if points.ndim != 3 or points.shape[-1] != 3:
    raise ValueError(f"Se esperaba points con shape (B, N, 3), pero se obtuvo {points.shape}")

total_clouds = points.shape[0]
num_extract = min(num_clouds, total_clouds)

print(f"\nTotal de nubes en NPZ: {total_clouds}")
print(f"Extrayendo primeras: {num_extract}")

# ==========================
# Guardar primeras nubes como XYZ
# ==========================

for i in range(num_extract):
    cloud = points[i]

    if "label" in data.files:
        label = data["label"][i]
    else:
        label = "unknown"

    if "category_name" in data.files:
        category_name = str(data["category_name"][i])
    else:
        category_name = "unknown"

    output_xyz = os.path.join(
        output_dir,
        f"pcd_{i:03d}_label_{label}_{category_name}.xyz"
    )

    np.savetxt(output_xyz, cloud, fmt="%.6f")

    print(f"Guardado: {output_xyz}")

# ==========================
# Guardar NPZ reducido opcional
# ==========================

if output_npz != "":
    output_dict = {}

    for k in data.files:
        arr = data[k]

        # Si la primera dimensión coincide con el número de nubes,
        # se recortan las primeras num_extract entradas.
        if hasattr(arr, "shape") and len(arr.shape) > 0 and arr.shape[0] == total_clouds:
            output_dict[k] = arr[:num_extract]
        else:
            output_dict[k] = arr

    output_npz_dir = os.path.dirname(output_npz)
    if output_npz_dir != "":
        os.makedirs(output_npz_dir, exist_ok=True)

    np.savez(output_npz, **output_dict)

    print(f"\nNPZ reducido guardado en:")
    print(output_npz)

    check = np.load(output_npz, allow_pickle=True)

    print("\nContenido del NPZ reducido:")
    for k in check.files:
        print(f"  {k}: shape = {check[k].shape}, dtype = {check[k].dtype}")

print("\nProceso terminado correctamente.")