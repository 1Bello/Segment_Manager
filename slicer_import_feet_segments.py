"""
slicer_import_feet_segments.py

Script para ejecutar dentro de 3D Slicer.
Convierte mascaras NIfTI (pie izquierdo/derecho) a segmentos en una
Segmentation y, opcionalmente, exporta STL.

Ejemplo (desde terminal):
  "C:/Program Files/Slicer 5.6.2/Slicer.exe" --no-main-window --python-script slicer_import_feet_segments.py -- \
    --ct "D:/3D slicer code/Impresion-Medicina/Pie/scan.nii.gz" \
    --left_mask "D:/3D slicer code/Impresion-Medicina/output_pie/foot_left_mask.nii.gz" \
    --right_mask "D:/3D slicer code/Impresion-Medicina/output_pie/foot_right_mask.nii.gz" \
    --out_seg "D:/3D slicer code/Impresion-Medicina/output_pie/feet.seg.nrrd" \
    --export_stl \
    --stl_dir "D:/3D slicer code/Impresion-Medicina/output_pie/stl"
"""

import argparse
import os
import sys

import slicer


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Importar mascaras de pie como segmentos en 3D Slicer")
    parser.add_argument("--ct", required=True, help="Ruta al volumen CT/CBCT (NIfTI)")
    parser.add_argument("--left_mask", required=True, help="Mascara pie izquierdo (NIfTI)")
    parser.add_argument("--right_mask", default=None, help="Mascara pie derecho (NIfTI)")
    parser.add_argument("--out_seg", required=True, help="Salida de segmentacion (.seg.nrrd)")
    parser.add_argument("--export_stl", action="store_true", help="Exportar STL de segmentos")
    parser.add_argument("--stl_dir", default=None, help="Directorio para STL (requerido si --export_stl)")
    return parser.parse_args(argv)


def ensure_file(path_value: str, label: str):
    if not os.path.exists(path_value):
        raise FileNotFoundError(f"No se encontro {label}: {path_value}")


def load_volume(path_value: str, what: str):
    loaded = slicer.util.loadVolume(path_value)
    if loaded is None:
        raise RuntimeError(f"No se pudo cargar {what}: {path_value}")
    return loaded


def load_labelmap(path_value: str, what: str):
    loaded = slicer.util.loadLabelVolume(path_value)
    if loaded is None:
        raise RuntimeError(f"No se pudo cargar {what}: {path_value}")
    return loaded


def import_mask_as_segment(segmentation_node, mask_path: str, segment_name: str):
    label_node = load_labelmap(mask_path, segment_name)
    logic = slicer.modules.segmentations.logic()

    existing_ids = set()
    seg = segmentation_node.GetSegmentation()
    for i in range(seg.GetNumberOfSegments()):
        existing_ids.add(seg.GetNthSegmentID(i))

    ok = logic.ImportLabelmapToSegmentationNode(label_node, segmentation_node)
    if not ok:
        raise RuntimeError(f"No se pudo importar mascara a segmentacion: {mask_path}")

    new_ids = []
    seg = segmentation_node.GetSegmentation()
    for i in range(seg.GetNumberOfSegments()):
        sid = seg.GetNthSegmentID(i)
        if sid not in existing_ids:
            new_ids.append(sid)

    if not new_ids:
        raise RuntimeError(f"No se detectaron segmentos nuevos al importar: {mask_path}")

    # Si la mascara es binaria normal, deberia crear solo 1 segmento.
    first_id = new_ids[0]
    seg.GetSegment(first_id).SetName(segment_name)

    slicer.mrmlScene.RemoveNode(label_node)


def main():
    args = parse_args()

    ensure_file(args.ct, "CT")
    ensure_file(args.left_mask, "mascara izquierda")
    if args.right_mask:
        ensure_file(args.right_mask, "mascara derecha")
    if args.export_stl and not args.stl_dir:
        raise ValueError("Debes indicar --stl_dir cuando usas --export_stl")

    print("[1/5] Cargando volumen CT...")
    ct_node = load_volume(args.ct, "CT")

    print("[2/5] Creando nodo de segmentacion...")
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "FeetSegmentation")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(ct_node)

    print("[3/5] Importando pie izquierdo...")
    import_mask_as_segment(segmentation_node, args.left_mask, "foot_left")

    if args.right_mask:
        print("[4/5] Importando pie derecho...")
        import_mask_as_segment(segmentation_node, args.right_mask, "foot_right")
    else:
        print("[4/5] Sin mascara derecha (omitido).")

    out_seg_dir = os.path.dirname(args.out_seg)
    if out_seg_dir:
        os.makedirs(out_seg_dir, exist_ok=True)

    print("[5/5] Guardando segmentacion...")
    if not slicer.util.saveNode(segmentation_node, args.out_seg):
        raise RuntimeError(f"No se pudo guardar segmentacion en: {args.out_seg}")
    print(f"[ok] Segmentacion guardada: {args.out_seg}")

    if args.export_stl:
        os.makedirs(args.stl_dir, exist_ok=True)
        segmentation_node.CreateClosedSurfaceRepresentation()
        logic = slicer.modules.segmentations.logic()
        ok = logic.ExportSegmentsClosedSurfaceRepresentationToFiles(
            args.stl_dir,
            segmentation_node,
            None,
            "STL",
        )
        if not ok:
            raise RuntimeError("Fallo la exportacion STL")
        print(f"[ok] STL exportado en: {args.stl_dir}")


if __name__ == "__main__":
    main()
