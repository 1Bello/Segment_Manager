"""
auto_segment.py  (v2 — optimizado para cuello y cabeza)
--------------------------------------------------------
Segmentación automática de escáneres de cuello/cabeza (NIfTI)
usando TotalSegmentator con tasks específicos para la región cervical.

INSTALACIÓN (una sola vez):
    pip install TotalSegmentator nibabel numpy

USO:
    python auto_segment.py --input escaner.nii.gz --output ./segmentaciones/
    python auto_segment.py --input escaner.nii.gz --output ./segmentaciones/ --fast
    python auto_segment.py --input escaner.nii.gz --output ./segmentaciones/ --cpu

Proyecto: Automatización impresión 3D fantomas médicos (traqueotomía)
Universidad de los Andes — Ignacio Bello & Tomás Rodríguez
Tutor: Jorge Gomez
"""

import argparse
import os
import sys
import time
import shutil
import nibabel as nib
import numpy as np

# ── Estructuras de interés para cuello/cabeza ─────────────────────────────────
# Solo estas se copian a la carpeta final. El resto se descarta.
ESTRUCTURAS_OBJETIVO = {
    # Vía aérea — prioridad máxima para traqueotomía
    "trachea":                  ("Tráquea",               (0,   174, 239)),
    "larynx_air":               ("Laringe (aire)",         (0,   210, 255)),

    # Glándulas y tejidos blandos
    "thyroid_gland":            ("Tiroides",               (255, 180,  50)),
    "esophagus":                ("Esófago",                (200,  80,  80)),

    # Vascular
    "carotid_artery_left":      ("Art. carótida izq.",     (220,  50,  50)),
    "carotid_artery_right":     ("Art. carótida der.",     (180,  30,  30)),
    "jugular_vein_left":        ("Vena yugular izq.",      ( 60,  60, 200)),
    "jugular_vein_right":       ("Vena yugular der.",      ( 40,  40, 160)),
    "brachiocephalic_trunk":    ("Tronco braquiocefálico", (160,  30,  30)),

    # Vértebras cervicales
    "vertebrae_C1":             ("Vértebra C1",            (220, 220, 180)),
    "vertebrae_C2":             ("Vértebra C2",            (210, 210, 170)),
    "vertebrae_C3":             ("Vértebra C3",            (200, 200, 160)),
    "vertebrae_C4":             ("Vértebra C4",            (190, 190, 150)),
    "vertebrae_C5":             ("Vértebra C5",            (180, 180, 140)),
    "vertebrae_C6":             ("Vértebra C6",            (170, 170, 130)),
    "vertebrae_C7":             ("Vértebra C7",            (160, 160, 120)),

    # Huesos de referencia
    "skull":                    ("Cráneo",                 (240, 230, 210)),
    "clavicula_left":           ("Clavícula izq.",         (200, 190, 170)),
    "clavicula_right":          ("Clavícula der.",         (195, 185, 165)),
    "sternum":                  ("Esternón",               (210, 200, 180)),
    "mandible":                 ("Mandíbula",              (230, 215, 190)),

    # Músculos
    "autochthon_left":          ("Músculo autóctono izq.", (180, 220, 180)),
    "autochthon_right":         ("Músculo autóctono der.", (160, 200, 160)),
}

# Tasks de TotalSegmentator a correr en secuencia para cuello/cabeza.
# Cada task aporta distintas estructuras; los resultados se fusionan.
TASKS_CUELLO = [
    "total",        # cubre tráquea, esófago, vascular, vértebras, huesos
    "vertebrae",    # mejor precisión en vértebras cervicales
    "head_glands_cavities",  # tiroides, laringe, glándulas
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Segmentación automática de cuello/cabeza con TotalSegmentator v2"
    )
    p.add_argument("--input",  "-i", required=True,
                   help="NIfTI de entrada (.nii o .nii.gz)")
    p.add_argument("--output", "-o", required=True,
                   help="Carpeta donde se guardan los segmentos finales")
    p.add_argument("--fast",  action="store_true",
                   help="Modo rápido (menor resolución, más velocidad)")
    p.add_argument("--cpu",   action="store_true",
                   help="Forzar CPU aunque haya GPU")
    return p.parse_args()


def verificar_input(ruta):
    if not os.path.exists(ruta):
        print(f"[ERROR] Archivo no encontrado: {ruta}")
        sys.exit(1)
    print(f"[OK] Entrada: {ruta}")


def correr_task(input_path, output_dir, task, fast, device):
    """Corre un task de TotalSegmentator y devuelve la carpeta con resultados."""
    from totalsegmentator.python_api import totalsegmentator

    task_dir = os.path.join(output_dir, f"_tmp_{task}")
    os.makedirs(task_dir, exist_ok=True)

    print(f"\n  → Task: {task} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        totalsegmentator(
            input=input_path,
            output=task_dir,
            task=task,
            fast=fast,
            device=device,
            quiet=True,
        )
        print(f"OK ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"ERROR: {e}")
        task_dir = None
    return task_dir


def fusionar_resultados(task_dirs, output_final):
    """
    Copia solo las estructuras de ESTRUCTURAS_OBJETIVO desde las carpetas
    temporales de cada task hacia output_final.
    Prioriza el primer task que tenga cada estructura (orden de TASKS_CUELLO).
    """
    os.makedirs(output_final, exist_ok=True)
    copiados = {}

    for task_dir in task_dirs:
        if task_dir is None or not os.path.isdir(task_dir):
            continue
        for nombre_estructura in ESTRUCTURAS_OBJETIVO:
            if nombre_estructura in copiados:
                continue  # ya fue copiada por un task anterior
            for ext in [".nii.gz", ".nii"]:
                src = os.path.join(task_dir, nombre_estructura + ext)
                if os.path.exists(src):
                    dst = os.path.join(output_final, nombre_estructura + ext)
                    shutil.copy2(src, dst)
                    copiados[nombre_estructura] = dst
                    break

    return copiados


def limpiar_temporales(output_dir):
    """Elimina carpetas temporales _tmp_* generadas por cada task."""
    for entry in os.listdir(output_dir):
        if entry.startswith("_tmp_"):
            ruta = os.path.join(output_dir, entry)
            shutil.rmtree(ruta, ignore_errors=True)


def filtrar_vacios(copiados):
    """Elimina archivos que no tienen ningún voxel segmentado."""
    validos = {}
    vacios = []
    for nombre, ruta in copiados.items():
        img = nib.load(ruta)
        if np.sum(img.get_fdata() > 0) > 0:
            validos[nombre] = ruta
        else:
            os.remove(ruta)
            vacios.append(nombre)
    return validos, vacios


def generar_reporte(output_dir, validos, vacios):
    ruta = os.path.join(output_dir, "reporte_segmentacion.txt")
    with open(ruta, "w") as f:
        f.write("REPORTE DE SEGMENTACIÓN — Cuello/Cabeza\n")
        f.write("TotalSegmentator v2 | Proyecto Fantomas Médicos\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Estructuras con contenido : {len(validos)}\n")
        f.write(f"Estructuras vacías        : {len(vacios)}\n\n")
        f.write("Segmentos exportados:\n")
        for nombre in validos:
            label, color = ESTRUCTURAS_OBJETIVO[nombre]
            f.write(f"  {nombre:<35} | {label:<30} | RGB{color}\n")
        if vacios:
            f.write("\nNo detectados en este escáner:\n")
            for nombre in vacios:
                label, _ = ESTRUCTURAS_OBJETIVO[nombre]
                f.write(f"  – {label}\n")
    print(f"[OK] Reporte: {ruta}")


def main():
    args = parse_args()
    verificar_input(args.input)
    os.makedirs(args.output, exist_ok=True)
    device = "cpu" if args.cpu else "gpu"

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Auto-Segmentador Cuello/Cabeza  v2         ║")
    print("║   Proyecto Fantomas / U. de los Andes        ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Dispositivo : {device.upper()}")
    print(f"  Modo rápido : {'Sí' if args.fast else 'No'}")
    print(f"  Tasks       : {', '.join(TASKS_CUELLO)}\n")

    # 1. Correr cada task
    task_dirs = []
    for task in TASKS_CUELLO:
        task_dir = correr_task(args.input, args.output, task, args.fast, device)
        task_dirs.append(task_dir)

    # 2. Fusionar — quedarse solo con estructuras de interés
    print("\n[INFO] Fusionando resultados...")
    copiados = fusionar_resultados(task_dirs, args.output)

    # 3. Filtrar vacíos
    validos, vacios = filtrar_vacios(copiados)

    # 4. Limpiar temporales
    limpiar_temporales(args.output)

    # 5. Reporte
    generar_reporte(args.output, validos, vacios)

    # 6. Resumen
    print("\n── Resultado ───────────────────────────────────────────────")
    print(f"  Estructuras detectadas : {len(validos)}")
    print(f"  No detectadas          : {len(vacios)}")
    print(f"  Carpeta de salida      : {os.path.abspath(args.output)}")
    print("────────────────────────────────────────────────────────────")
    print("\n  Siguiente paso:")
    print("  Corre cargar_segmentos_slicer.py en la consola de 3D Slicer")
    print(f"  apuntando a: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
