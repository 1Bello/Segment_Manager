"""
eval_medsam.py
Instalación, inferencia y evaluación de MedSAM contra ground truth del radiólogo.

MedSAM es un modelo prompt-based (SAM fine-tuneado en imágenes médicas).
Requiere bounding boxes 2D como prompt por slice. Este script genera los
bounding boxes automáticamente desde el ground truth para una evaluación justa
(modo "upper-bound semi-automático").

Uso:
    python eval_medsam.py --ct ruta/al/scan.nii.gz --gt_dir ruta/ground_truth/ --out resultados_medsam.json

Estructura esperada de gt_dir:
    ground_truth/
        trachea.nii.gz
        vertebrae_C1.nii.gz
        thyroid_gland.nii.gz
        ...  (un archivo por estructura, nombrado con el nombre de la estructura)

Nota: MedSAM opera slice a slice en 2D. Este script procesa el eje axial (z).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# ── Instalación automática de dependencias ─────────────────────────────────────

def install_dependencies():
    pkgs = [
        "nibabel", "numpy", "scipy", "Pillow",
        "torch", "torchvision",
        "git+https://github.com/bowang-lab/MedSAM.git",
    ]
    print("[setup] Instalando dependencias de MedSAM...")
    # torch primero (puede ser grande)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "-q"])
    for pkg in ["nibabel", "numpy", "scipy", "Pillow"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    # MedSAM desde GitHub
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "git+https://github.com/bowang-lab/MedSAM.git", "-q"])
    print("[setup] Dependencias listas.\n")

try:
    import nibabel as nib
    import torch
    from segment_anything import sam_model_registry
except ImportError:
    install_dependencies()
    import nibabel as nib
    import torch
    from segment_anything import sam_model_registry


# ── Descarga del checkpoint de MedSAM ─────────────────────────────────────────

MEDSAM_CHECKPOINT_URL = "https://zenodo.org/record/7966557/files/medsam_vit_b.pth"
MEDSAM_CHECKPOINT_PATH = Path("medsam_vit_b.pth")

def download_checkpoint():
    if MEDSAM_CHECKPOINT_PATH.exists():
        print(f"[setup] Checkpoint ya existe: {MEDSAM_CHECKPOINT_PATH}")
        return
    print(f"[setup] Descargando checkpoint MedSAM (~375 MB)...")
    print(f"        URL: {MEDSAM_CHECKPOINT_URL}")
    urllib.request.urlretrieve(MEDSAM_CHECKPOINT_URL, MEDSAM_CHECKPOINT_PATH)
    print("[setup] Checkpoint descargado.\n")


# ── Estructuras de cuello a evaluar ───────────────────────────────────────────

NECK_STRUCTURES_GT = [
    "trachea", "thyroid_gland", "esophagus",
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
    "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "parotid_gland_left", "parotid_gland_right",
    "submandibular_gland_left", "submandibular_gland_right",
    "larynx_air",
]


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


# ── Preprocesamiento de slice para SAM ────────────────────────────────────────

def preprocess_slice(ct_slice: np.ndarray, win_min=-200, win_max=300) -> np.ndarray:
    """
    Windowing CT → [0,255] uint8 RGB (SAM espera imágenes RGB).
    Ventana por defecto: tejido blando (-200 a 300 HU).
    """
    slc = np.clip(ct_slice, win_min, win_max)
    slc = ((slc - win_min) / (win_max - win_min) * 255).astype(np.uint8)
    return np.stack([slc, slc, slc], axis=-1)  # (H, W, 3)


def get_bbox_from_mask(mask_slice: np.ndarray, margin: int = 5) -> list[int] | None:
    """Extrae bounding box 2D de una máscara binaria con margen."""
    rows = np.any(mask_slice, axis=1)
    cols = np.any(mask_slice, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask_slice.shape
    return [
        max(0, int(cmin) - margin),
        max(0, int(rmin) - margin),
        min(W, int(cmax) + margin),
        min(H, int(rmax) + margin),
    ]


# ── Inferencia MedSAM slice a slice ───────────────────────────────────────────

@torch.no_grad()
def medsam_inference_slice(medsam_model, img_embed, bbox: list[int], H: int, W: int):
    """Corre el decoder de MedSAM dado un embedding de imagen y un bounding box."""
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(1024)

    box_1024 = np.array(bbox)
    box_1024[[0, 2]] = box_1024[[0, 2]] / W * 1024
    box_1024[[1, 3]] = box_1024[[1, 3]] / H * 1024
    box_torch = torch.as_tensor(box_1024[None, None, :],
                                dtype=torch.float, device=medsam_model.device)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    # Upscale a resolución original
    low_res_pred = torch.sigmoid(low_res_logits)
    from torch.nn import functional as F
    upscaled = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    return (upscaled[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def run_medsam(ct_path: Path, gt_dir: Path, checkpoint: Path, use_gpu: bool) -> tuple[list[dict], float]:
    """
    Corre MedSAM en modo semi-automático: usa el bounding box del GT para guiar
    la segmentación. Retorna resultados por estructura y tiempo total.
    """
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[MedSAM] Cargando modelo en {device}...")

    # Cargar modelo
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()
    medsam_model = sam

    ct_img  = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()
    voxel_sizes = ct_img.header.get_zooms()[:3]

    results = []
    t_start = time.time()

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
        }

        if gt_path is None:
            print(f"  {structure}: sin GT → omitido")
            results.append(entry)
            continue

        gt_img  = nib.load(str(gt_path))
        gt_data = gt_img.get_fdata()

        if gt_data.shape != ct_data.shape:
            print(f"  [WARN] {structure}: shape GT {gt_data.shape} ≠ CT {ct_data.shape} → omitido")
            results.append(entry)
            continue

        # Segmentar slice a slice en eje axial
        pred_volume = np.zeros_like(gt_data, dtype=np.uint8)
        n_slices_with_gt = 0

        for z in range(ct_data.shape[2]):
            gt_slice = gt_data[:, :, z]
            bbox = get_bbox_from_mask(gt_slice)
            if bbox is None:
                continue

            n_slices_with_gt += 1
            ct_slice = ct_data[:, :, z]
            H, W = ct_slice.shape
            img_rgb = preprocess_slice(ct_slice)

            # Resize a 1024x1024 para SAM
            from PIL import Image
            pil_img = Image.fromarray(img_rgb).resize((1024, 1024))
            img_tensor = torch.as_tensor(np.array(pil_img).transpose(2, 0, 1),
                                         dtype=torch.float32, device=device).unsqueeze(0) / 255.0

            # Normalización SAM
            pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            pixel_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            with torch.no_grad():
                img_embed = medsam_model.image_encoder(img_tensor)

            pred_slice = medsam_inference_slice(medsam_model, img_embed, bbox, H, W)
            pred_volume[:, :, z] = pred_slice

        entry["prediccion_encontrada"] = n_slices_with_gt > 0
        entry["dice"] = round(dice_score(pred_volume, gt_data), 4)
        entry["iou"]  = round(iou_score(pred_volume, gt_data), 4)
        entry["volumen_pred_ml"] = round(volume_ml(pred_volume, voxel_sizes[:3]), 2)
        entry["volumen_gt_ml"]   = round(volume_ml(gt_data, gt_img.header.get_zooms()[:3]), 2)

        print(f"  {structure}: Dice={entry['dice']}  IoU={entry['iou']}  slices_GT={n_slices_with_gt}")
        results.append(entry)

    elapsed = time.time() - t_start
    return results, elapsed


# ── Reporte ────────────────────────────────────────────────────────────────────

def build_report(results: list[dict], elapsed: float, ct_path: Path, use_gpu: bool) -> dict:
    evaluados = [r for r in results if r["dice"] is not None]
    dices = [r["dice"] for r in evaluados]
    ious  = [r["iou"]  for r in evaluados]

    return {
        "modelo": "MedSAM (SAM fine-tuned médico — modo semi-automático con bbox GT)",
        "nota_metodologica": (
            "Los bounding boxes se extrajeron del ground truth (modo upper-bound). "
            "En producción, MedSAM requiere prompts manuales o una etapa de detección previa."
        ),
        "scan": str(ct_path),
        "dispositivo": "GPU" if use_gpu else "CPU",
        "tiempo_inferencia": {
            "total_segundos": round(elapsed, 2),
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
    parser = argparse.ArgumentParser(description="Evaluación MedSAM vs ground truth")
    parser.add_argument("--ct",         required=True,  help="Ruta al scan CT (.nii o .nii.gz)")
    parser.add_argument("--gt_dir",     required=True,  help="Directorio con segmentaciones manuales")
    parser.add_argument("--out",        default="resultados_medsam.json", help="Archivo JSON de salida")
    parser.add_argument("--checkpoint", default=str(MEDSAM_CHECKPOINT_PATH), help="Ruta al checkpoint .pth de MedSAM")
    parser.add_argument("--cpu",        action="store_true", help="Forzar uso de CPU")
    args = parser.parse_args()

    ct_path    = Path(args.ct)
    gt_dir     = Path(args.gt_dir)
    checkpoint = Path(args.checkpoint)

    assert ct_path.exists(), f"No se encontró el CT: {ct_path}"
    assert gt_dir.is_dir(),  f"No se encontró el directorio GT: {gt_dir}"

    use_gpu = not args.cpu

    # Descargar checkpoint si no existe
    if not checkpoint.exists():
        download_checkpoint()
        checkpoint = MEDSAM_CHECKPOINT_PATH

    print("=" * 60)
    print("  EVALUACIÓN: MedSAM")
    print(f"  CT:         {ct_path}")
    print(f"  GT dir:     {gt_dir}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device:     {'GPU' if use_gpu else 'CPU'}")
    print("=" * 60)

    results, elapsed = run_medsam(ct_path, gt_dir, checkpoint, use_gpu)
    report = build_report(results, elapsed, ct_path, use_gpu)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Reporte guardado en: {out_path}")
    print(f"     Dice promedio: {report['metricas_agregadas']['dice_promedio']}")
    print(f"     Estructuras comparadas: {report['estructuras']['comparadas_exitosamente']} / {report['estructuras']['total_evaluadas']}")
    print(f"     Tiempo total: {report['tiempo_inferencia']['total_minutos']} min")


if __name__ == "__main__":
    main()
