"""
cargar_segmentos_slicer.py  (v2 — compatible Slicer 5.10)
----------------------------------------------------------
Carga segmentos NIfTI en 3D Slicer con:
  - Representación 3D activada desde el inicio (surface rendering)
  - Nombres en español y colores por estructura
  - Compatible con Slicer 5.10 (corrige error threeDViewNames)

CÓMO USARLO:
    1. Abre 3D Slicer
    2. View → Python Interactor  (o Ctrl+3)
    3. exec(open("/Users/trodriguezam/Universidad/Titulo/Segment_Manager/load_auto_segment.py").read())

Proyecto: Automatización impresión 3D fantomas médicos (traqueotomía)
Universidad de los Andes — Ignacio Bello & Tomás Rodríguez
Tutor: Jorge Gomez
"""

import os
import slicer
import numpy as np

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
# Cambia esta ruta para no tener que ingresarla cada vez.
# Déjala vacía "" para que el script te la pida al correr.
CARPETA_SEGMENTOS = "/Users/trodriguezam/Universidad/Titulo/Segment_Manager/total_segmentator_output"

# Limpia la escena antes de cargar
LIMPIAR_ESCENA = True
# ──────────────────────────────────────────────────────────────────────────────


ESTRUCTURAS = {
    "trachea":                  ("Tráquea",               (0,   174, 239)),
    "larynx_air":               ("Laringe (aire)",         (0,   210, 255)),
    "thyroid_gland":            ("Tiroides",               (255, 180,  50)),
    "esophagus":                ("Esófago",                (200,  80,  80)),
    "carotid_artery_left":      ("Art. carótida izq.",     (220,  50,  50)),
    "carotid_artery_right":     ("Art. carótida der.",     (180,  30,  30)),
    "jugular_vein_left":        ("Vena yugular izq.",      ( 60,  60, 200)),
    "jugular_vein_right":       ("Vena yugular der.",      ( 40,  40, 160)),
    "brachiocephalic_trunk":    ("Tronco braquiocefálico", (160,  30,  30)),
    "vertebrae_C1":             ("Vértebra C1",            (220, 220, 180)),
    "vertebrae_C2":             ("Vértebra C2",            (210, 210, 170)),
    "vertebrae_C3":             ("Vértebra C3",            (200, 200, 160)),
    "vertebrae_C4":             ("Vértebra C4",            (190, 190, 150)),
    "vertebrae_C5":             ("Vértebra C5",            (180, 180, 140)),
    "vertebrae_C6":             ("Vértebra C6",            (170, 170, 130)),
    "vertebrae_C7":             ("Vértebra C7",            (160, 160, 120)),
    "skull":                    ("Cráneo",                 (240, 230, 210)),
    "clavicula_left":           ("Clavícula izq.",         (200, 190, 170)),
    "clavicula_right":          ("Clavícula der.",         (195, 185, 165)),
    "sternum":                  ("Esternón",               (210, 200, 180)),
    "mandible":                 ("Mandíbula",              (230, 215, 190)),
    "autochthon_left":          ("Músculo autóctono izq.", (180, 220, 180)),
    "autochthon_right":         ("Músculo autóctono der.", (160, 200, 160)),
}


def obtener_carpeta():
    carpeta = CARPETA_SEGMENTOS.strip()
    if not carpeta:
        carpeta = input(
            "\nRuta a la carpeta con los segmentos .nii.gz\n> "
        ).strip().strip('"').strip("'")
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(f"Carpeta no encontrada: {carpeta}")
    return carpeta


def listar_niftis(carpeta):
    return sorted([
        f for f in os.listdir(carpeta)
        if (f.endswith(".nii.gz") or f.endswith(".nii"))
        and not f.startswith("reporte")
    ])


def nombre_desde_archivo(archivo):
    return archivo.replace(".nii.gz", "").replace(".nii", "")


def rgb01(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0


def esta_vacio(ruta):
    # import nibabel as nib
    # img = nib.load(ruta)
    # return np.sum(img.get_fdata() > 0) == 0
    pass


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Cargador de Segmentos v2 — Slicer 5.10     ║")
    print("║   Proyecto Fantomas / U. de los Andes        ║")
    print("╚══════════════════════════════════════════════╝\n")

    carpeta = obtener_carpeta()
    print(f"[OK] Carpeta: {carpeta}")

    if LIMPIAR_ESCENA:
        slicer.mrmlScene.Clear(0)
        print("[OK] Escena limpiada.")

    archivos = listar_niftis(carpeta)
    if not archivos:
        print("[ERROR] No se encontraron archivos .nii.gz en la carpeta.")
        return
    print(f"[INFO] Archivos encontrados: {len(archivos)}")

    # Crear nodo de segmentación contenedor
    nombre_caso = os.path.basename(carpeta.rstrip("/\\")) or "Cuello"
    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    seg_node.SetName(nombre_caso)
    seg_node.CreateDefaultDisplayNodes()

    cargados = []
    omitidos = []

    for i, archivo in enumerate(archivos):
        nombre = nombre_desde_archivo(archivo)
        ruta   = os.path.join(carpeta, archivo)

        # Obtener label y color
        if nombre in ESTRUCTURAS:
            label, color_rgb = ESTRUCTURAS[nombre]
        else:
            label    = nombre
            color_rgb = (128, 128, 128)

        print(f"  [{i+1}/{len(archivos)}] {label}...", end=" ", flush=True)

        # Saltar vacíos sin ni siquiera cargarlos en la escena
        if esta_vacio(ruta):
            print("(vacío, omitido)")
            omitidos.append(label)
            continue

        try:
            # Cargar como labelmap temporal
            labelmap = slicer.util.loadLabelVolume(ruta)

            # Importar al nodo de segmentación
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap, seg_node
            )

            # El segmento recién agregado es el último
            seg      = seg_node.GetSegmentation()
            segmento = seg.GetNthSegment(seg.GetNumberOfSegments() - 1)

            # Nombre y color
            segmento.SetName(label)
            segmento.SetColor(*rgb01(*color_rgb))

            # Eliminar labelmap temporal
            slicer.mrmlScene.RemoveNode(labelmap)

            print("✓")
            cargados.append(label)

        except Exception as e:
            print(f"ERROR: {e}")
            omitidos.append(label)

    # ── CRÍTICO: activar representación 3D (surface) ─────────────────────────
    # Sin esto los segmentos aparecen en los slices pero desaparecen en la
    # vista 3D al mover la cámara.
    print("\n[INFO] Generando superficies 3D...", end=" ", flush=True)
    seg_node.CreateClosedSurfaceRepresentation()
    print("✓")

    # ── Centrar vistas (compatible Slicer 5.10) ───────────────────────────────
    try:
        # Vista 3D
        lm = slicer.app.layoutManager()
        for i in range(lm.threeDViewCount):
            lm.threeDWidget(i).threeDView().resetFocalPoint()
        # Vistas de slices
        slicer.util.resetSliceViews()
        print("[OK] Vistas centradas.")
    except Exception as e:
        print(f"[AVISO] No se pudo centrar la vista: {e}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n── Resumen ─────────────────────────────────────────────────")
    print(f"  Cargados : {len(cargados)}")
    print(f"  Omitidos : {len(omitidos)}")
    if cargados:
        print("\n  Estructuras en escena:")
        for l in cargados:
            print(f"    ✓ {l}")
    if omitidos:
        print("\n  No detectadas en este escáner:")
        for l in omitidos:
            print(f"    – {l}")
    print("────────────────────────────────────────────────────────────\n")
    print("Listo. Puedes girar la vista 3D libremente.")


main()
