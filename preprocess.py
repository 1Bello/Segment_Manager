"""
preprocesar_ct.py
-----------------
Normaliza un escáner CT antes de pasarlo a TotalSegmentator.
Hace clip de valores extremos (artefactos fuera del cuerpo) que
confunden al modelo de segmentación.

USO:
    python preprocesar_ct.py --input caso.nii.gz --output caso_norm.nii.gz

Proyecto: Automatización impresión 3D fantomas médicos (traqueotomía)
Universidad de los Andes — Ignacio Bello & Tomás Rodríguez
Tutor: Jorge Gomez
"""

import argparse
import numpy as np
import nibabel as nib

# Rango estándar CT en Hounsfield Units
# Todo lo que esté fuera de este rango es artefacto o padding del scanner
CT_MIN = -1024
CT_MAX = 3071


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Cargando: {args.input}")
    img = nib.load(args.input)
    arr = img.get_fdata()

    print(f"[INFO] Rango original  : min={arr.min():.0f}  max={arr.max():.0f}")

    # Clip al rango CT estándar
    arr_clip = np.clip(arr, CT_MIN, CT_MAX)

    print(f"[INFO] Rango después   : min={arr_clip.min():.0f}  max={arr_clip.max():.0f}")

    # Guardar con el mismo header y affine originales
    img_out = nib.Nifti1Image(arr_clip, img.affine, img.header)
    nib.save(img_out, args.output)

    print(f"[OK] Guardado en: {args.output}")
    print("\n  Siguiente paso:")
    print(f"  python auto_segment_v2.py --input {args.output} --output ./segmentaciones/")


if __name__ == "__main__":
    main()
