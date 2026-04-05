"""
eval_totalsegmentator.py
Instalación, inferencia y evaluación de TotalSegmentator contra ground truth del radiólogo.

Uso:
    python eval_totalsegmentator.py --ct ruta/al/scan.nii.gz --gt_dir ruta/ground_truth/ --out resultados_ts.json

Estructura esperada de gt_dir:
    ground_truth/
        trachea.nii.gz
        vertebrae_C1.nii.gz
        thyroid_gland.nii.gz
        ...  (un archivo por estructura, nombrado con el nombre de la estructura)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Instalación automática de dependencias ─────────────────────────────────────

def install_dependencies():
    pkgs = ["totalsegmentator", "nibabel", "numpy", "scipy"]
    print("[setup] Instalando dependencias...")
    for pkg in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("[setup] Dependencias listas.\n")

try:
    import nibabel as nib
except ImportError:
    install_dependencies()
    import nibabel as nib


# ── Estructuras de cuello relevantes para el proyecto ─────────────────────────
# TotalSegmentator usa estos nombres internamente (snake_case).
# Se busca coincidencia flexible con los archivos del ground truth.

NECK_STRUCTURES = [
    # task: total
    "trachea",
    "thyroid_gland",
    "esophagus",
    "sternum",
    "heart",
    # task: vertebrae
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
    "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    # task: head_glands_cavities
    "parotid_gland_left", "parotid_gland_right",
    "submandibular_gland_left", "submandibular_gland_right",
    "nasal_cavity_left", "nasal_cavity_right",
    "larynx_air", "skull",
]

TASKS = ["total", "vertebrae", "head_glands_cavities"]


# ── Métricas ───────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt   > 0).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    if denom == 0:
        return 1.0  # ambos vacíos → coincidencia perfecta
    return float(2 * intersection / denom)

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt   > 0).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 1.0
    return float(intersection / union)

def volume_ml(mask: np.ndarray, voxel_sizes: tuple) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_sizes))
    return float(np.sum(mask > 0) * voxel_vol_mm3 / 1000.0)


# ── Matching flexible entre nombres de estructura ──────────────────────────────

def match_gt_file(structure_name: str, gt_dir: Path) -> Path | None:
    """
    Busca en gt_dir el archivo cuyo nombre (sin extensión) coincida con
    structure_name de forma flexible (case-insensitive, ignorando separadores).
    """
    candidates = list(gt_dir.glob("*.nii")) + list(gt_dir.glob("*.nii.gz"))
    norm = lambda s: s.lower().replace("-", "_").replace(" ", "_")
    target = norm(structure_name)
    for c in candidates:
        stem = c.name.replace(".nii.gz", "").replace(".nii", "")
        if norm(stem) == target:
            return c
    # intento parcial
    for c in candidates:
        stem = c.name.replace(".nii.gz", "").replace(".nii", "")
        if target in norm(stem) or norm(stem) in target:
            return c
    return None


# ── Inferencia TotalSegmentator ────────────────────────────────────────────────

def run_totalsegmentator(ct_path: Path, output_dir: Path, use_gpu: bool) -> dict:
    """
    Ejecuta los 3 tasks relevantes para cuello via CLI (más estable entre versiones).
    Retorna dict con tiempos por task.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tiempos = {}

    for task in TASKS:
        task_out = output_dir / task
        task_out.mkdir(exist_ok=True)
        print(f"[TotalSegmentator] Corriendo task '{task}'...")

        cmd = [
            sys.executable, "-m", "totalsegmentator",
            "-i", str(ct_path),
            "-o", str(task_out),
            "--task", task,
        ]
        if use_gpu:
            cmd += ["--device", "gpu"]
        else:
            cmd += ["--device", "cpu"]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [WARN] Task '{task}' terminó con errores:")
            print(result.stderr[-1500:])  # últimas líneas del error
        else:
            print(f"  → Task '{task}' completado en {elapsed:.1f}s")

        tiempos[task] = round(elapsed, 2)

    return tiempos


# ── Evaluación contra ground truth ────────────────────────────────────────────

def evaluate(output_dir: Path, gt_dir: Path) -> list[dict]:
    results = []

    for structure in NECK_STRUCTURES:
        # Buscar predicción en alguno de los task dirs
        pred_path = None
        for task in TASKS:
            candidate = output_dir / task / f"{structure}.nii.gz"
            if candidate.exists():
                pred_path = candidate
                break

        gt_path = match_gt_file(structure, gt_dir)

        entry = {
            "estructura": structure,
            "prediccion_encontrada": pred_path is not None,
            "gt_encontrado": gt_path is not None,
            "dice": None,
            "iou": None,
            "volumen_pred_ml": None,
            "volumen_gt_ml": None,
        }

        if pred_path and gt_path:
            pred_img = nib.load(str(pred_path))
            gt_img   = nib.load(str(gt_path))
            pred_arr = pred_img.get_fdata()
            gt_arr   = gt_img.get_fdata()

            # Verificar compatibilidad de forma
            if pred_arr.shape != gt_arr.shape:
                print(f"  [WARN] {structure}: shapes distintos pred={pred_arr.shape} gt={gt_arr.shape} — se omite Dice")
            else:
                entry["dice"] = round(dice_score(pred_arr, gt_arr), 4)
                entry["iou"]  = round(iou_score(pred_arr, gt_arr), 4)

            vox_pred = pred_img.header.get_zooms()[:3]
            vox_gt   = gt_img.header.get_zooms()[:3]
            entry["volumen_pred_ml"] = round(volume_ml(pred_arr, vox_pred), 2)
            entry["volumen_gt_ml"]   = round(volume_ml(gt_arr,   vox_gt),   2)

        results.append(entry)
        status = f"Dice={entry['dice']}" if entry["dice"] is not None else "sin coincidencia GT"
        print(f"  {structure}: {status}")

    return results


# ── Reporte resumen ────────────────────────────────────────────────────────────

def build_report(tiempos: dict, results: list[dict], ct_path: Path, use_gpu: bool) -> dict:
    estructuras_detectadas = [r for r in results if r["prediccion_encontrada"]]
    estructuras_con_gt     = [r for r in results if r["gt_encontrado"]]
    estructuras_evaluadas  = [r for r in results if r["dice"] is not None]

    dices = [r["dice"] for r in estructuras_evaluadas]
    ious  = [r["iou"]  for r in estructuras_evaluadas]

    return {
        "modelo": "TotalSegmentator (nnU-Net)",
        "scan": str(ct_path),
        "dispositivo": "GPU" if use_gpu else "CPU",
        "tiempo_inferencia": {
            **tiempos,
            "total_segundos": sum(tiempos.values()),
            "total_minutos": round(sum(tiempos.values()) / 60, 2),
        },
        "estructuras": {
            "total_evaluadas": len(NECK_STRUCTURES),
            "detectadas_por_modelo": len(estructuras_detectadas),
            "con_gt_disponible": len(estructuras_con_gt),
            "comparadas_exitosamente": len(estructuras_evaluadas),
        },
        "metricas_agregadas": {
            "dice_promedio": round(np.mean(dices), 4) if dices else None,
            "dice_mediana":  round(float(np.median(dices)), 4) if dices else None,
            "iou_promedio":  round(np.mean(ious), 4) if ious else None,
            "dice_min":      round(min(dices), 4) if dices else None,
            "dice_max":      round(max(dices), 4) if dices else None,
        },
        "detalle_por_estructura": results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluación TotalSegmentator vs ground truth")
    parser.add_argument("--ct",      required=True,  help="Ruta al scan CT (.nii o .nii.gz)")
    parser.add_argument("--gt_dir",  required=True,  help="Directorio con segmentaciones manuales (.nii.gz por estructura)")
    parser.add_argument("--out",     default="resultados_totalsegmentator.json", help="Archivo JSON de salida")
    parser.add_argument("--out_dir", default="ts_output", help="Directorio donde guardar segmentaciones predichas")
    parser.add_argument("--cpu",     action="store_true", help="Forzar uso de CPU (por defecto usa GPU si está disponible)")
    args = parser.parse_args()

    ct_path  = Path(args.ct)
    gt_dir   = Path(args.gt_dir)
    out_path = Path(args.out)
    out_dir  = Path(args.out_dir)

    assert ct_path.exists(),  f"No se encontró el CT: {ct_path}"
    assert gt_dir.is_dir(),   f"No se encontró el directorio GT: {gt_dir}"

    use_gpu = not args.cpu

    print("=" * 60)
    print("  EVALUACIÓN: TotalSegmentator")
    print(f"  CT:      {ct_path}")
    print(f"  GT dir:  {gt_dir}")
    print(f"  Device:  {'GPU' if use_gpu else 'CPU'}")
    print("=" * 60)

    # 1. Inferencia
    tiempos = run_totalsegmentator(ct_path, out_dir, use_gpu)

    # 2. Evaluación
    print("\n[eval] Calculando métricas contra ground truth...")
    results = evaluate(out_dir, gt_dir)

    # 3. Reporte
    report = build_report(tiempos, results, ct_path, use_gpu)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Reporte guardado en: {out_path}")
    print(f"     Dice promedio: {report['metricas_agregadas']['dice_promedio']}")
    print(f"     Estructuras comparadas: {report['estructuras']['comparadas_exitosamente']} / {report['estructuras']['total_evaluadas']}")
    print(f"     Tiempo total: {report['tiempo_inferencia']['total_minutos']} min")


if __name__ == "__main__":
    main()
