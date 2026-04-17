"""
segment_foot_tissues.py

Separa una mascara de pie en:
1) Hueso (bone_mask)
2) Tejido blando (soft_tissue_mask)

Estrategia:
- Trabaja solo dentro de la mascara del pie.
- Hueso: umbral de intensidad (HU) + limpieza morfologica.
- Tejido blando: pie - hueso.

Uso:
    python segment_foot_tissues.py \
      --ct D:/ruta/ct.nii.gz \
      --foot_mask D:/ruta/foot_left_mask.nii.gz \
      --out_dir D:/ruta/output_left
"""

import argparse
from pathlib import Path

import numpy as np


def install_dependencies() -> None:
    import subprocess
    import sys

    for pkg in ["nibabel", "numpy", "scipy"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from scipy import ndimage
except ImportError:
    install_dependencies()
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from scipy import ndimage


def keep_largest_components(mask: np.ndarray, max_components: int = 20, min_voxels: int = 200) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, nlab = ndimage.label(mask.astype(np.uint8), structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    items = []
    for lab in range(1, nlab + 1):
        size = int(np.count_nonzero(labeled == lab))
        if size >= min_voxels:
            items.append((lab, size))

    if not items:
        return np.zeros_like(mask, dtype=np.uint8)

    items.sort(key=lambda x: x[1], reverse=True)
    keep_labels = {lab for lab, _ in items[:max_components]}
    return np.isin(labeled, list(keep_labels)).astype(np.uint8)


def segment_bone_and_soft_tissue(
    ct_data: np.ndarray,
    foot_mask: np.ndarray,
    bone_threshold: float,
    close_iter: int,
    open_iter: int,
    max_components: int,
    min_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    foot_bin = (foot_mask > 0).astype(np.uint8)

    # Hueso por umbral dentro de la ROI del pie, evitando temporales gigantes.
    bone = np.zeros_like(foot_bin, dtype=np.uint8)
    foot_idx = np.where(foot_bin > 0)
    if foot_idx[0].size > 0:
        bone[foot_idx] = (ct_data[foot_idx] >= bone_threshold).astype(np.uint8)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    if open_iter > 0:
        bone = ndimage.binary_opening(bone, structure=structure, iterations=open_iter).astype(np.uint8)
    if close_iter > 0:
        bone = ndimage.binary_closing(bone, structure=structure, iterations=close_iter).astype(np.uint8)

    bone = keep_largest_components(bone, max_components=max_components, min_voxels=min_voxels)
    soft = foot_bin.copy()
    bone_idx = np.where(bone > 0)
    if bone_idx[0].size > 0:
        soft[bone_idx] = 0
    return bone, soft


def volume_ml(mask: np.ndarray, voxel_sizes: tuple[float, float, float]) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    return float(np.count_nonzero(mask) * voxel_vol_mm3 / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Separar hueso y tejido blando dentro de una mascara de pie")
    parser.add_argument("--ct", required=True, help="Ruta CT/CBCT en NIfTI")
    parser.add_argument("--foot_mask", required=True, help="Mascara del pie en NIfTI")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida")
    parser.add_argument("--bone_threshold", type=float, default=200.0, help="Umbral para hueso (default 200)")
    parser.add_argument("--open_iter", type=int, default=1, help="Iteraciones opening (default 1)")
    parser.add_argument("--close_iter", type=int, default=1, help="Iteraciones closing (default 1)")
    parser.add_argument("--max_components", type=int, default=20, help="Max. componentes oseos a conservar")
    parser.add_argument("--min_voxels", type=int, default=200, help="Tamano minimo por componente")
    parser.add_argument(
        "--no_auto_resample",
        action="store_true",
        help="Desactiva remuestreo automatico de la mascara cuando no coincide con CT",
    )
    args = parser.parse_args()

    ct_path = Path(args.ct)
    foot_mask_path = Path(args.foot_mask)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ct_path.exists():
        raise FileNotFoundError(f"No se encontro CT: {ct_path}")
    if not foot_mask_path.exists():
        raise FileNotFoundError(f"No se encontro mascara de pie: {foot_mask_path}")

    ct_img = nib.load(str(ct_path))
    foot_img = nib.load(str(foot_mask_path))
    foot_data = foot_img.get_fdata(dtype=np.float32)

    output_ref_img = foot_img

    if ct_img.shape != foot_img.shape:
        if args.no_auto_resample:
            raise ValueError(f"Shape distinto CT={ct_img.shape} mask={foot_img.shape}")

        print(f"[warn] Shape distinto CT={ct_img.shape} mask={foot_img.shape}")
        print("[prep] Remuestreando CT al espacio de la mascara (mas eficiente en memoria)...")
        # order=1 para intensidades continuas del CT.
        ct_img = resample_from_to(ct_img, foot_img, order=1)
        output_ref_img = foot_img

        resampled_ct_path = out_dir / "ct_resampled_to_mask.nii.gz"
        nib.save(ct_img, str(resampled_ct_path))
        print(f"[info] CT remuestreado guardado en: {resampled_ct_path}")
    else:
        output_ref_img = ct_img

    ct_data = ct_img.get_fdata(dtype=np.float32)

    bone_mask, soft_mask = segment_bone_and_soft_tissue(
        ct_data=ct_data,
        foot_mask=foot_data,
        bone_threshold=args.bone_threshold,
        close_iter=args.close_iter,
        open_iter=args.open_iter,
        max_components=args.max_components,
        min_voxels=args.min_voxels,
    )

    bone_path = out_dir / "bone_mask.nii.gz"
    soft_path = out_dir / "soft_tissue_mask.nii.gz"

    # Mantener la geometria del espacio de trabajo (CT original o mascara si hubo remuestreo).
    nib.save(nib.Nifti1Image(bone_mask.astype(np.uint8), output_ref_img.affine, output_ref_img.header), str(bone_path))
    nib.save(nib.Nifti1Image(soft_mask.astype(np.uint8), output_ref_img.affine, output_ref_img.header), str(soft_path))

    voxel_sizes = output_ref_img.header.get_zooms()[:3]
    print("[ok] Archivos generados:")
    print(f"  - {bone_path}")
    print(f"  - {soft_path}")
    print("[info] Volumenes (ml):")
    print(f"  - Bone: {volume_ml(bone_mask, voxel_sizes):.2f}")
    print(f"  - Soft tissue: {volume_ml(soft_mask, voxel_sizes):.2f}")


if __name__ == "__main__":
    main()
