"""
segmentar_cuello.py
====================
Segmentación automática de cabeza, cuello y hombros usando TotalSegmentator.
Las estructuras objetivo están basadas en la segmentación manual del radiólogo
(ground truth del proyecto).

Cambios respecto a versión anterior:
  - Se eliminó 'vertebrae_body' (requiere licencia de pago).
    Las vértebras C1-C7 se obtienen desde el task 'total'.
  - Se agrega filtro estricto: solo se copian a segmentos_finales/ los
    archivos cuyo nombre esté en SEGMENTOS_RADIOLOGO (o en MAPA_NOMBRES).
    Todo lo demás (glúteos, piernas, etc.) se descarta silenciosamente.

Uso:
    python segmentar_cuello.py --input ruta/scan.nii.gz --output ruta/salida/

Requisitos:
    pip install totalsegmentator nibabel numpy
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH: lista exacta de segmentos del radiólogo
# ─────────────────────────────────────────────────────────────────────────────

SEGMENTOS_RADIOLOGO = [
    "anterior_scalene_left", "anterior_scalene_right",
    "auditory_canal_left", "auditory_canal_right",
    "body", "body_extremities", "body_trunc",
    "eye_left", "eye_lens_left", "eye_lens_right", "eye_right",
    "hard_palate",
    "head",
    "hypopharynx",
    "inferior_pharyngeal_constrictor",
    "levator_scapulae_left", "levator_scapulae_right",
    "mandible",
    "middle_pharyngeal_constrictor",
    "middle_scalene_left", "middle_scalene_right",
    "nasal_cavity_left", "nasal_cavity_right",
    "nasopharynx",
    "optic_nerve_left", "optic_nerve_right",
    "oropharynx",
    "parotid_gland_left", "parotid_gland_right",
    "platysma_left", "platysma_right",
    "posterior_scalene_left", "posterior_scalene_right",
    "prevertebral_left", "prevertebral_right",
    "sinus_frontal", "sinus_maxillary",
    "skeletal_muscle",
    "skin",
    "skull",
    "soft_palate",
    "sterno_thyroid_left", "sterno_thyroid_right",
    "sternocleidomastoid_left", "sternocleidomastoid_right",
    "subcutaneous_fat",
    "submandibular_gland_left", "submandibular_gland_right",
    "superior_pharyngeal_constrictor",
    "teeth_lower", "teeth_upper",
    "thyrohyoid_left", "thyrohyoid_right",
    "torso_fat",
    "trachea",
    "trapezius_left", "trapezius_right",
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
    "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
]

# Set para lookup O(1)
SEGMENTOS_RADIOLOGO_SET = set(SEGMENTOS_RADIOLOGO)

# ─────────────────────────────────────────────────────────────────────────────
# MAPEO: nombre TotalSegmentator → nombre radiólogo
# Solo para estructuras donde el nombre difiere entre herramientas.
# ─────────────────────────────────────────────────────────────────────────────

MAPA_NOMBRES = {
    "sternocleido_left":       "sternocleidomastoid_left",
    "sternocleido_right":      "sternocleidomastoid_right",
    "scalenus_anterior_left":  "anterior_scalene_left",
    "scalenus_anterior_right": "anterior_scalene_right",
    "scalenus_medius_left":    "middle_scalene_left",
    "scalenus_medius_right":   "middle_scalene_right",
    "scalenus_posterior_left": "posterior_scalene_left",
    "scalenus_posterior_right":"posterior_scalene_right",
}

# Nombres válidos finales = union de ambas fuentes
NOMBRES_VALIDOS = SEGMENTOS_RADIOLOGO_SET | set(MAPA_NOMBRES.keys())

# Tasks a correr (sin vertebrae_body — requiere licencia)
TASKS = [
    "total",                # tráquea, skull, mandible, parotid, submandibular, C1-C7, vasos
    "head_glands_cavities", # nasal cavity, sinuses, eye, optic nerve, nasopharynx, oropharynx
    "body",                 # body, body_trunc, body_extremities, head, sternocleidomastoid, trapezius
    "tissue_types",         # skin, subcutaneous_fat, skeletal_muscle, torso_fat
]

# Categorías para el reporte
CATEGORIAS_GT = {
    "vertebras": [
        "vertebrae_C1","vertebrae_C2","vertebrae_C3","vertebrae_C4",
        "vertebrae_C5","vertebrae_C6","vertebrae_C7",
    ],
    "oseas": [
        "skull","mandible","hard_palate","teeth_upper","teeth_lower",
    ],
    "musculos": [
        "sternocleidomastoid_left","sternocleidomastoid_right",
        "trapezius_left","trapezius_right",
        "anterior_scalene_left","anterior_scalene_right",
        "middle_scalene_left","middle_scalene_right",
        "posterior_scalene_left","posterior_scalene_right",
        "levator_scapulae_left","levator_scapulae_right",
        "platysma_left","platysma_right",
        "sterno_thyroid_left","sterno_thyroid_right",
        "thyrohyoid_left","thyrohyoid_right",
        "prevertebral_left","prevertebral_right",
        "inferior_pharyngeal_constrictor",
        "middle_pharyngeal_constrictor",
        "superior_pharyngeal_constrictor",
        "skeletal_muscle",
    ],
    "via_aerea": [
        "trachea","nasopharynx","oropharynx","hypopharynx","soft_palate",
        "nasal_cavity_left","nasal_cavity_right",
        "sinus_frontal","sinus_maxillary",
    ],
    "glandulas": [
        "parotid_gland_left","parotid_gland_right",
        "submandibular_gland_left","submandibular_gland_right",
    ],
    "ojos": [
        "eye_left","eye_right","eye_lens_left","eye_lens_right",
        "optic_nerve_left","optic_nerve_right",
        "auditory_canal_left","auditory_canal_right",
    ],
    "tejidos_blandos": ["skin","subcutaneous_fat","torso_fat"],
    "cuerpo":          ["body","body_trunc","body_extremities","head"],
}


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE GPU
# ─────────────────────────────────────────────────────────────────────────────

def detectar_device(device_arg: str) -> str:
    """
    Verifica si CUDA está disponible. Si el usuario pidió GPU pero no hay,
    advierte y cae a CPU en lugar de fallar silenciosamente.
    """
    if device_arg == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            nombre = torch.cuda.get_device_name(0)
            print(f"  → GPU detectada: {nombre}")
            return "gpu"
        else:
            print("  ⚠ GPU no disponible (CUDA no detectado). Usando CPU.")
            print("    Si tienes GPU, verifica que el entorno Python tiene")
            print("    el torch correcto: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return "cpu"
    except ImportError:
        print("  ⚠ torch no encontrado. Usando CPU.")
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_ct(input_path: Path, output_dir: Path) -> Path:
    """
    Recorta valores HU fuera del rango estándar CT (-1024, 3071).
    Los artefactos de padding del escáner degradan la segmentación.
    """
    print("  → Preprocesando CT (recorte de HU outliers)...")
    img = nib.load(str(input_path))
    data = img.get_fdata()

    v_min, v_max = float(data.min()), float(data.max())
    data_clip = np.clip(data, -1024, 3071)

    out_path = output_dir / "ct_preprocesado.nii.gz"
    nib.save(nib.Nifti1Image(data_clip, img.affine, img.header), str(out_path))

    print(f"    HU original:  [{v_min:.0f}, {v_max:.0f}]")
    print(f"    HU recortado: [-1024, 3071]")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def correr_task(input_path: Path, output_dir: Path, task: str, device: str) -> dict:
    task_dir = output_dir / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ▶ [{task}]")
    t0 = time.time()
    try:
        totalsegmentator(
            input=str(input_path),
            output=str(task_dir),
            task=task,
            device=device,
            fastest=False,
            quiet=False,
            verbose=False,
        )
        dur = time.time() - t0
        archivos = list(task_dir.glob("*.nii.gz"))
        nombres = [f.stem.replace(".nii", "") for f in archivos]
        relevantes = [n for n in nombres if n in NOMBRES_VALIDOS]
        print(f"    ✓ {len(archivos)} segmentos generados, {len(relevantes)} relevantes — {dur:.0f}s")
        return {
            "task": task, "status": "ok",
            "duracion_s": round(dur, 1),
            "n_total": len(archivos),
            "n_relevantes": len(relevantes),
        }
    except Exception as e:
        dur = time.time() - t0
        print(f"    ✗ Error: {e}")
        return {"task": task, "status": "error", "duracion_s": round(dur, 1), "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLIDACIÓN CON FILTRO ESTRICTO
#
# Solo se copian a segmentos_finales/ archivos cuyo nombre (original o
# traducido por MAPA_NOMBRES) esté en SEGMENTOS_RADIOLOGO_SET.
# Todo lo demás (glúteos, fémures, pulmones, etc.) se descarta.
# ─────────────────────────────────────────────────────────────────────────────

def consolidar(output_dir: Path) -> dict:
    finales = output_dir / "segmentos_finales"
    finales.mkdir(exist_ok=True)

    copiados = {}       # nombre_final → path origen
    descartados = []    # nombres que no pasaron el filtro

    tasks_dir = output_dir / "tasks"
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for archivo in task_dir.glob("*.nii.gz"):
            nombre_ts = archivo.stem.replace(".nii", "")

            # Traducir si hay alias
            nombre_final = MAPA_NOMBRES.get(nombre_ts, nombre_ts)

            # FILTRO: descartar si no está en el ground truth
            if nombre_final not in SEGMENTOS_RADIOLOGO_SET:
                if nombre_ts not in descartados:
                    descartados.append(nombre_ts)
                continue

            # Copiar solo si no existe ya (primer task que lo genera gana)
            if nombre_final not in copiados:
                destino = finales / f"{nombre_final}.nii.gz"
                shutil.copy2(str(archivo), str(destino))
                copiados[nombre_final] = str(archivo)

    detectados    = sorted(copiados.keys())
    no_detectados = sorted(SEGMENTOS_RADIOLOGO_SET - set(copiados.keys()))

    print(f"    Descartados (fuera de GT): {len(descartados)} segmentos")

    return {
        "detectados":         detectados,
        "no_detectados":      no_detectados,
        "n_generados_total":  len(copiados) + len(no_detectados),
        "n_descartados":      len(descartados),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────────────────────────────────────

def generar_reporte(output_dir: Path, resultados_tasks: list,
                    consolidacion: dict, t_total: float):
    detectados_set = set(consolidacion["detectados"])

    cobertura_cat = {}
    for cat, estructuras in CATEGORIAS_GT.items():
        det = [e for e in estructuras if e in detectados_set]
        cobertura_cat[cat] = {
            "detectadas": len(det),
            "total":      len(estructuras),
            "pct":        round(100 * len(det) / len(estructuras), 1),
            "faltantes":  [e for e in estructuras if e not in detectados_set],
        }

    n_gt       = len(SEGMENTOS_RADIOLOGO)
    n_det      = len(consolidacion["detectados"])
    pct_global = round(100 * n_det / n_gt, 1)

    reporte = {
        "resumen": {
            "cobertura_global_pct":    pct_global,
            "detectadas_vs_radiologo": f"{n_det}/{n_gt}",
            "segmentos_descartados":   consolidacion["n_descartados"],
            "tiempo_total_min":        round(t_total / 60, 1),
        },
        "cobertura_por_categoria": cobertura_cat,
        "tareas":        resultados_tasks,
        "no_detectados": consolidacion["no_detectados"],
    }

    reporte_path = output_dir / "reporte_segmentacion.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 62)
    print("  REPORTE DE SEGMENTACIÓN")
    print("═" * 62)
    print(f"  Cobertura global:   {n_det}/{n_gt} estructuras del radiólogo ({pct_global}%)")
    print(f"  Descartados:        {consolidacion['n_descartados']} segmentos fuera del GT")
    print(f"  Tiempo total:       {round(t_total/60,1)} min")
    print()
    print("  Cobertura por categoría:")
    for cat, datos in cobertura_cat.items():
        bloques = int(datos["pct"] / 10)
        barra   = "█" * bloques + "░" * (10 - bloques)
        print(f"    {cat:<22} {barra}  {datos['detectadas']}/{datos['total']} ({datos['pct']}%)")

    if consolidacion["no_detectados"]:
        print(f"\n  No detectadas ({len(consolidacion['no_detectados'])}):")
        for e in consolidacion["no_detectados"]:
            print(f"    - {e}")

    print(f"\n  Reporte JSON:  {reporte_path}")
    print(f"  Segmentos:     {output_dir / 'segmentos_finales'}")
    print("═" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segmentación de cabeza/cuello/hombros con TotalSegmentator"
    )
    parser.add_argument("--input",           required=True, help="Archivo NIfTI (.nii / .nii.gz)")
    parser.add_argument("--output",          required=True, help="Directorio de salida")
    parser.add_argument("--device",          default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Omitir preprocesamiento HU")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

    print("\n" + "═" * 62)
    print("  SEGMENTACIÓN CUELLO / CABEZA / HOMBROS")
    print("═" * 62)
    print(f"  Input:    {input_path.name}")
    print(f"  Output:   {output_path}")
    print(f"  Tasks:    {', '.join(TASKS)}")
    print(f"  GT ref:   {len(SEGMENTOS_RADIOLOGO)} segmentos del radiólogo")
    print("═" * 62)

    # Verificar GPU
    device = detectar_device(args.device)

    t_inicio = time.time()

    # 1. Preprocesamiento
    ct_path = input_path
    if not args.skip_preprocess:
        ct_path = preprocesar_ct(input_path, output_path)
    else:
        print("  → Preprocesamiento omitido.")

    # 2. Correr tasks
    print("\n── Segmentación ──────────────────────────────────────────")
    resultados_tasks = [correr_task(ct_path, output_path, t, device) for t in TASKS]

    # 3. Consolidar con filtro
    print("\n── Consolidando (filtrando por GT del radiólogo) ─────────")
    consolidacion = consolidar(output_path)
    print(f"  → {len(consolidacion['detectados'])}/{len(SEGMENTOS_RADIOLOGO)} estructuras del radiólogo en segmentos_finales/")

    # 4. Reporte
    t_total = time.time() - t_inicio
    generar_reporte(output_path, resultados_tasks, consolidacion, t_total)


if __name__ == "__main__":
    main()