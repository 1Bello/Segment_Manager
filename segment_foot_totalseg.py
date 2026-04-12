"""
segment_foot_totalseg.py

Script independiente para segmentar pie(s) usando TotalSegmentator cuando
solo se tiene el volumen completo (sin ground truth).

Estrategia:
1) Ejecuta TotalSegmentator con task "body".
2) Usa la máscara body_extremities como punto de partida.
3) Conserva componentes conectados en la región más inferior del eje SI.
4) Guarda máscara combinada de pies y separación izquierda/derecha.

Uso con NIfTI:
    python segment_foot_totalseg.py --ct ruta/al/scan.nii.gz --out_dir salida_pie

Uso con DICOM:
    python segment_foot_totalseg.py --dicom_dir ruta/a/carpeta_dicom --out_dir salida_pie
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def install_dependencies() -> None:
    pkgs = ["totalsegmentator", "nibabel", "numpy", "scipy", "SimpleITK"]
    print("[setup] Instalando dependencias...")
    for pkg in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("[setup] Dependencias listas.\n")


try:
    import nibabel as nib
    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator as ts_api
    from scipy import ndimage
except ImportError:
    install_dependencies()
    import nibabel as nib
    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator as ts_api
    from scipy import ndimage


def run_totalsegmentator_body(ct_path: Path, output_dir: Path, use_gpu: bool) -> Path:
    body_dir = output_dir / "task_body"
    body_dir.mkdir(parents=True, exist_ok=True)

    print("[TotalSegmentator] Ejecutando task 'body'...")
    t0 = time.time()
    ts_api(
        input=str(ct_path),
        output=str(body_dir),
        task="body",
        device="gpu" if use_gpu else "cpu",
        output_type="nifti",
        quiet=False,
        verbose=False,
    )
    elapsed = time.time() - t0

    print(f"[TotalSegmentator] Completado en {elapsed / 60:.2f} min")
    return body_dir


def convert_dicom_to_nifti(dicom_dir: Path, out_dir: Path) -> Path:
    """Convierte una serie DICOM a NIfTI usando SimpleITK."""
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"No se encontró directorio DICOM: {dicom_dir}")

    print("[prep] Convirtiendo DICOM a NIfTI...")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError("No se encontraron series DICOM en el directorio indicado.")

    # Usa la primera serie detectada. Si hubiera más de una, se puede extender luego.
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not file_names:
        raise RuntimeError("No se pudieron listar los archivos de la serie DICOM.")

    reader.SetFileNames(file_names)
    image = reader.Execute()

    nifti_path = out_dir / "input_from_dicom.nii.gz"
    sitk.WriteImage(image, str(nifti_path))
    print(f"[prep] NIfTI generado: {nifti_path}")
    return nifti_path


def get_axis_index(axcodes: tuple[str, str, str], pair: tuple[str, str]) -> int:
    for i, c in enumerate(axcodes):
        if c in pair:
            return i
    raise ValueError(f"No se encontró eje {pair} en orientación {axcodes}")


def inferior_slice_selector(indices: np.ndarray, si_axis: int, si_code: str, keep_percent: float) -> np.ndarray:
    """Selecciona voxeles de la zona más inferior del volumen para aislar pies."""
    si_vals = indices[:, si_axis]
    if si_code == "S":
        # Si +index apunta a superior, inferior está en índices bajos
        threshold = np.percentile(si_vals, keep_percent)
        return si_vals <= threshold
    # Si +index apunta a inferior, inferior está en índices altos
    threshold = np.percentile(si_vals, 100 - keep_percent)
    return si_vals >= threshold


def extract_feet_from_extremities(
    extremities_mask: np.ndarray,
    axcodes: tuple[str, str, str],
    keep_inferior_percent: float = 20.0,
    max_components: int = 2,
    min_component_voxels: int = 500,
) -> np.ndarray:
    mask = (extremities_mask > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        raise ValueError("La máscara body_extremities está vacía.")

    si_axis = get_axis_index(axcodes, ("S", "I"))
    si_code = axcodes[si_axis]

    coords = np.argwhere(mask > 0)
    candidate_flags = inferior_slice_selector(coords, si_axis, si_code, keep_inferior_percent)
    candidates = np.zeros_like(mask, dtype=np.uint8)
    candidates[tuple(coords[candidate_flags].T)] = 1

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        raise ValueError("No se pudieron etiquetar componentes conectados.")

    chosen = []
    for lab in range(1, nlab + 1):
        comp = labeled == lab
        comp_size = int(np.count_nonzero(comp))
        if comp_size < min_component_voxels:
            continue
        overlap = int(np.count_nonzero(comp & (candidates > 0)))
        if overlap > 0:
            chosen.append((lab, comp_size, overlap))

    if not chosen:
        raise ValueError("No se encontraron componentes de extremidades en la zona inferior.")

    chosen.sort(key=lambda x: (x[2], x[1]), reverse=True)
    selected_labels = {lab for lab, _, _ in chosen[:max_components]}

    feet_mask = np.isin(labeled, list(selected_labels)).astype(np.uint8)
    return feet_mask


def split_left_right(mask: np.ndarray, axcodes: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray]:
    lr_axis = get_axis_index(axcodes, ("L", "R"))
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        empty = np.zeros_like(mask, dtype=np.uint8)
        return empty, empty

    midpoint = int(np.median(coords[:, lr_axis]))
    left = np.zeros_like(mask, dtype=np.uint8)
    right = np.zeros_like(mask, dtype=np.uint8)

    left_idx = coords[coords[:, lr_axis] <= midpoint]
    right_idx = coords[coords[:, lr_axis] > midpoint]
    if left_idx.size > 0:
        left[tuple(left_idx.T)] = 1
    if right_idx.size > 0:
        right[tuple(right_idx.T)] = 1

    return left, right


def volume_ml(mask: np.ndarray, voxel_sizes: tuple[float, float, float]) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    return float(np.count_nonzero(mask) * voxel_vol_mm3 / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentación de pie(s) usando TotalSegmentator")
    parser.add_argument("--ct", default=None, help="Ruta al volumen CT (.nii o .nii.gz)")
    parser.add_argument("--dicom_dir", default=None, help="Directorio con serie DICOM")
    parser.add_argument("--out_dir", default="foot_ts_output", help="Directorio de salida")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    parser.add_argument(
        "--inferior_percent",
        type=float,
        default=20.0,
        help="Porcentaje inferior del eje SI para buscar pies (default 20)",
    )
    parser.add_argument("--skip_inference", action="store_true", help="Usar salida existente de TotalSegmentator")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ct and not args.dicom_dir:
        raise ValueError("Debes indicar --ct o --dicom_dir.")
    if args.ct and args.dicom_dir:
        raise ValueError("Usa solo una entrada: --ct o --dicom_dir.")

    if args.ct:
        ct_path = Path(args.ct)
        if not ct_path.exists():
            raise FileNotFoundError(f"No se encontró CT: {ct_path}")
    else:
        dicom_dir = Path(args.dicom_dir)
        ct_path = convert_dicom_to_nifti(dicom_dir, out_dir)

    use_gpu = not args.cpu

    print("=" * 70)
    print("  SEGMENTACIÓN DE PIE(S) CON TOTALSEGMENTATOR")
    print(f"  CT/NIfTI usado: {ct_path}")
    print(f"  Salida: {out_dir}")
    print(f"  Dispositivo: {'GPU' if use_gpu else 'CPU'}")
    print("=" * 70)

    if args.skip_inference:
        body_dir = out_dir / "task_body"
        if not body_dir.exists():
            raise FileNotFoundError("No existe task_body y pediste --skip_inference")
    else:
        body_dir = run_totalsegmentator_body(ct_path, out_dir, use_gpu)

    extremities_path = body_dir / "body_extremities.nii.gz"
    if not extremities_path.exists():
        raise FileNotFoundError(
            f"No se encontró {extremities_path}. Verifica que task 'body' terminó correctamente."
        )

    ext_img = nib.load(str(extremities_path))
    ext_data = ext_img.get_fdata()
    axcodes = nib.orientations.aff2axcodes(ext_img.affine)

    feet_mask = extract_feet_from_extremities(
        extremities_mask=ext_data,
        axcodes=axcodes,
        keep_inferior_percent=float(args.inferior_percent),
    )
    foot_left, foot_right = split_left_right(feet_mask, axcodes)

    feet_path = out_dir / "feet_mask.nii.gz"
    left_path = out_dir / "foot_left_mask.nii.gz"
    right_path = out_dir / "foot_right_mask.nii.gz"

    nib.save(nib.Nifti1Image(feet_mask.astype(np.uint8), ext_img.affine, ext_img.header), str(feet_path))
    nib.save(nib.Nifti1Image(foot_left.astype(np.uint8), ext_img.affine, ext_img.header), str(left_path))
    nib.save(nib.Nifti1Image(foot_right.astype(np.uint8), ext_img.affine, ext_img.header), str(right_path))

    voxel_sizes = ext_img.header.get_zooms()[:3]
    v_feet = volume_ml(feet_mask, voxel_sizes)
    v_left = volume_ml(foot_left, voxel_sizes)
    v_right = volume_ml(foot_right, voxel_sizes)

    print("\n[ok] Máscaras guardadas:")
    print(f"  - {feet_path}")
    print(f"  - {left_path}")
    print(f"  - {right_path}")
    print("\n[info] Volúmenes estimados (ml):")
    print(f"  Pie(s) total: {v_feet:.2f}")
    print(f"  Pie izquierdo: {v_left:.2f}")
    print(f"  Pie derecho: {v_right:.2f}")
    print(f"  Orientación detectada: {axcodes}")


if __name__ == "__main__":
    main()
