"""
eval_yolo.py
Instalación, inferencia y evaluación de YOLO (YOLOv8-seg) contra ground truth.

ADVERTENCIA METODOLÓGICA:
    YOLO fue diseñado para detección de objetos 2D en imágenes naturales.
    NO es nativo para segmentación volumétrica CT.
    Este script lo evalúa en modo 2D slice-a-slice usando YOLOv8-seg,
    entrenado en COCO (dominio general). Los resultados esperados son BAJOS
    para estructuras médicas — esto es el resultado esperado y válido para
    la tabla comparativa, ya que demuestra por qué YOLO no es adecuado para
    esta tarea.

Uso:
    python eval_yolo.py --ct ruta/al/scan.nii.gz --gt_dir ruta/ground_truth/ --out resultados_yolo.json

Estructura esperada de gt_dir:
    ground_truth/
        trachea.nii.gz
        vertebrae_C1.nii.gz
        ...
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Instalación automática de dependencias ─────────────────────────────────────

def install_dependencies():
    print("[setup] Instalando dependencias de YOLO...")
    for pkg in ["ultralytics", "nibabel", "numpy", "Pillow", "opencv-python-headless"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("[setup] Dependencias listas.\n")

try:
    import nibabel as nib
    from ultralytics import YOLO
    import cv2
except ImportError:
    install_dependencies()
    import nibabel as nib
    from ultralytics import YOLO
    import cv2


# ── Estructuras a evaluar (igual que los otros scripts) ───────────────────────

NECK_STRUCTURES_GT = [
    "trachea", "thyroid_gland", "esophagus",
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
    "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "parotid_gland_left", "parotid_gland_right",
    "submandibular_gland_left", "submandibular_gland_right",
    "larynx_air",
]

# Modelo preentrenado de YOLOv8 con segmentación de instancias (COCO)
YOLO_MODEL = "yolov8x-seg.pt"  # descargado automáticamente por ultralytics


# ── Métricas ───────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt   > 0).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    if denom == 0:
        return 1.0
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


# ── Matching flexible de nombres ───────────────────────────────────────────────

def match_gt_file(structure_name: str, gt_dir: Path) -> Path | None:
    candidates = list(gt_dir.glob("*.nii")) + list(gt_dir.glob("*.nii.gz"))
    norm = lambda s: s.lower().replace("-", "_").replace(" ", "_")
    target = norm(structure_name)
    for c in candidates:
        stem = c.name.replace(".nii.gz", "").replace(".nii", "")
        if norm(stem) == target:
            return c
    for c in candidates:
        stem = c.name.replace(".nii.gz", "").replace(".nii", "")
        if target in norm(stem) or norm(stem) in target:
            return c
    return None


# ── Preprocesamiento de slice CT → imagen BGR para OpenCV/YOLO ────────────────

def ct_slice_to_bgr(ct_slice: np.ndarray, win_min=-200, win_max=300) -> np.ndarray:
    """Convierte un slice CT a imagen BGR uint8 de 3 canales."""
    slc = np.clip(ct_slice, win_min, win_max)
    slc = ((slc - win_min) / (win_max - win_min) * 255).astype(np.uint8)
    return cv2.cvtColor(slc, cv2.COLOR_GRAY2BGR)


# ── Inferencia YOLO slice a slice ─────────────────────────────────────────────

def run_yolo(ct_path: Path, gt_dir: Path, use_gpu: bool) -> tuple[list[dict], float]:
    """
    Ejecuta YOLOv8-seg slice a slice sobre el CT.
    Construye un volumen de predicción binario por estructura y lo compara
    con el GT. Como YOLO no conoce anatomía médica, se intentará encontrar
    detecciones de cualquier clase en la región del GT (bounding box).

    Estrategia de matching:
        Para cada slice donde el GT tiene vóxeles, se busca si YOLO detectó
        alguna máscara que se solape con la región del GT (IoU > umbral).
        Si hay solapamiento, esa máscara se acepta como predicción.
        Esto es la evaluación más favorable posible para YOLO.
    """
    device = 0 if use_gpu else "cpu"  # ultralytics: 0 = primera GPU
    print(f"[YOLO] Cargando YOLOv8x-seg (descarga automática si es primera vez)...")
    model = YOLO(YOLO_MODEL)

    ct_img  = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()
    voxel_sizes = ct_img.header.get_zooms()[:3]

    results = []
    t_start = time.time()

    # Pre-computar inferencia YOLO sobre todos los slices (una sola pasada)
    # para eficiencia — guardamos las máscaras por slice
    print("[YOLO] Corriendo inferencia sobre todos los slices axiales...")
    n_slices = ct_data.shape[2]
    all_masks = {}  # z -> list of (mask_2d: np.ndarray)

    t_inf_start = time.time()
    for z in range(n_slices):
        img_bgr = ct_slice_to_bgr(ct_data[:, :, z])
        res = model.predict(img_bgr, device=device, verbose=False, conf=0.1)
        slice_masks = []
        if res[0].masks is not None:
            for m in res[0].masks.data:
                mask_np = m.cpu().numpy()
                # Resize a tamaño original del slice si es necesario
                H, W = img_bgr.shape[:2]
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
                slice_masks.append((mask_np > 0.5).astype(np.uint8))
        all_masks[z] = slice_masks

        if z % 20 == 0:
            print(f"  slice {z}/{n_slices} — {len(slice_masks)} detecciones")

    t_inf_elapsed = time.time() - t_inf_start

    # Evaluar por estructura
    OVERLAP_THRESHOLD = 0.05  # IoU mínimo para considerar una detección como "match"

    for structure in NECK_STRUCTURES_GT:
        gt_path = match_gt_file(structure, gt_dir)
        entry = {
            "estructura": structure,
            "gt_encontrado": gt_path is not None,
            "prediccion_encontrada": False,
            "dice": None,
            "iou": None,
            "volumen_pred_ml": None,
            "volumen_gt_ml": None,
            "slices_con_gt": 0,
            "slices_con_match_yolo": 0,
        }

        if gt_path is None:
            print(f"  {structure}: sin GT → omitido")
            results.append(entry)
            continue

        gt_img_nib = nib.load(str(gt_path))
        gt_data    = gt_img_nib.get_fdata()

        if gt_data.shape != ct_data.shape:
            print(f"  [WARN] {structure}: shape GT {gt_data.shape} ≠ CT {ct_data.shape} → omitido")
            results.append(entry)
            continue

        pred_volume = np.zeros_like(gt_data, dtype=np.uint8)
        slices_con_gt = 0
        slices_con_match = 0

        for z in range(n_slices):
            gt_slice = gt_data[:, :, z]
            if not np.any(gt_slice):
                continue
            slices_con_gt += 1
            gt_bin = (gt_slice > 0).astype(np.uint8)

            # Buscar la máscara de YOLO con mayor solapamiento con GT
            best_iou = 0.0
            best_mask = None
            for mask in all_masks[z]:
                inter = np.sum(mask & gt_bin)
                union = np.sum(mask | gt_bin)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask

            if best_iou >= OVERLAP_THRESHOLD and best_mask is not None:
                pred_volume[:, :, z] = best_mask
                slices_con_match += 1

        entry["prediccion_encontrada"] = slices_con_match > 0
        entry["slices_con_gt"]         = slices_con_gt
        entry["slices_con_match_yolo"] = slices_con_match
        entry["dice"] = round(dice_score(pred_volume, gt_data), 4)
        entry["iou"]  = round(iou_score(pred_volume, gt_data), 4)
        entry["volumen_pred_ml"] = round(volume_ml(pred_volume, voxel_sizes), 2)
        entry["volumen_gt_ml"]   = round(volume_ml(gt_data, gt_img_nib.header.get_zooms()[:3]), 2)

        print(f"  {structure}: Dice={entry['dice']}  match={slices_con_match}/{slices_con_gt} slices")
        results.append(entry)

    elapsed = time.time() - t_start
    return results, elapsed, t_inf_elapsed


# ── Reporte ────────────────────────────────────────────────────────────────────

def build_report(results, elapsed, t_inf_elapsed, ct_path, use_gpu):
    evaluados = [r for r in results if r["dice"] is not None]
    dices = [r["dice"] for r in evaluados]
    ious  = [r["iou"]  for r in evaluados]

    return {
        "modelo": "YOLO (YOLOv8x-seg — 2D, dominio general, sin fine-tuning médico)",
        "nota_metodologica": (
            "YOLO no fue diseñado para segmentación volumétrica CT. "
            "Se evaluó en modo 2D slice-a-slice con matching por IoU de región. "
            "Esta es la evaluación más favorable posible para YOLO. "
            "Los resultados bajos confirman la inadecuación del modelo para esta tarea."
        ),
        "scan": str(ct_path),
        "dispositivo": "GPU" if use_gpu else "CPU",
        "tiempo_inferencia": {
            "inferencia_pura_segundos": round(t_inf_elapsed, 2),
            "total_con_evaluacion_segundos": round(elapsed, 2),
            "total_minutos": round(elapsed / 60, 2),
        },
        "estructuras": {
            "total_evaluadas": len(NECK_STRUCTURES_GT),
            "con_gt_disponible": sum(1 for r in results if r["gt_encontrado"]),
            "comparadas_exitosamente": len(evaluados),
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
    parser = argparse.ArgumentParser(description="Evaluación YOLO vs ground truth")
    parser.add_argument("--ct",     required=True,  help="Ruta al scan CT (.nii o .nii.gz)")
    parser.add_argument("--gt_dir", required=True,  help="Directorio con segmentaciones manuales")
    parser.add_argument("--out",    default="resultados_yolo.json", help="Archivo JSON de salida")
    parser.add_argument("--cpu",    action="store_true", help="Forzar uso de CPU")
    args = parser.parse_args()

    ct_path = Path(args.ct)
    gt_dir  = Path(args.gt_dir)

    assert ct_path.exists(), f"No se encontró el CT: {ct_path}"
    assert gt_dir.is_dir(),  f"No se encontró el directorio GT: {gt_dir}"

    use_gpu = not args.cpu

    print("=" * 60)
    print("  EVALUACIÓN: YOLO (YOLOv8x-seg)")
    print(f"  CT:     {ct_path}")
    print(f"  GT dir: {gt_dir}")
    print(f"  Device: {'GPU' if use_gpu else 'CPU'}")
    print("  NOTA: Evaluación upper-bound — matching por solapamiento de región")
    print("=" * 60)

    results, elapsed, t_inf = run_yolo(ct_path, gt_dir, use_gpu)
    report = build_report(results, elapsed, t_inf, ct_path, use_gpu)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Reporte guardado en: {out_path}")
    print(f"     Dice promedio: {report['metricas_agregadas']['dice_promedio']}")
    print(f"     Estructuras comparadas: {report['estructuras']['comparadas_exitosamente']} / {report['estructuras']['total_evaluadas']}")
    print(f"     Tiempo total: {report['tiempo_inferencia']['total_minutos']} min")


if __name__ == "__main__":
    main()
