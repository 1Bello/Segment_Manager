import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ======================================================
# ✅ Obtener centro de un segmento en RAS
# ======================================================
def get_segment_center(segmentationNode, segmentName):
    segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    poly = vtk.vtkPolyData()
    segmentationNode.GetClosedSurfaceRepresentation(segmentID, poly)

    bounds = [0]*6
    poly.GetBounds(bounds)
    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ])
    return center


# ======================================================
# ✅ Corte usando plano RAS correctamente convertido a IJK
# ======================================================
def apply_cut(arr_xyz, rasToIjk, origin_ras, normal_ras):

    # Convertir plano a IJK
    origin_ijk = np.array(rasToIjk.MultiplyDoublePoint(list(origin_ras)+[1])[:3])
    normal_ijk = np.array(normal_ras)

    # Obtener dimensiones correctas
    X, Y, Z = arr_xyz.shape

    xx, yy, zz = np.meshgrid(
        np.arange(X),
        np.arange(Y),
        np.arange(Z),
        indexing='ij'
    )
    coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)

    # Aplicar corte
    diff = coords - origin_ijk
    mask = np.sum(diff * normal_ijk, axis=-1) < 0
    arr_xyz[mask] = 0

    return arr_xyz


# ======================================================
# ✅ Cargar segmentación
# ======================================================
case_name = "CASO_CUELLO_6"
segmentationNode = slicer.util.getNode(case_name)

# ======================================================
# ✅ Exportar a labelmap UNA VEZ
# ======================================================
labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "temp_labelmap")
slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapNode)

image = labelmapNode.GetImageData()
dims = image.GetDimensions()

# IMPORTANTE: convertir a XYZ
arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims, order="F")  # (x,y,z)

# Matriz RAS→IJK
rasToIjk = vtk.vtkMatrix4x4()
labelmapNode.GetRASToIJKMatrix(rasToIjk)

# ======================================================
# ✅ Obtener centro de tráquea
# ======================================================
trachea_center = get_segment_center(segmentationNode, "trachea")
print("Trachea center RAS:", trachea_center)

# ======================================================
# ✅ Definir cortes ANATÓMICOS REALISTAS
# ======================================================
cuts = [
    ("Cabeza", trachea_center + np.array([0, 0, 70]), [0, 0, -1]),
    ("Espalda", trachea_center + np.array([0, -50, 0]), [0, -1, 0]),
    ("Derecha", trachea_center + np.array([70, 0, 0]), [1, 0, 0]),
    ("Izquierda", trachea_center + np.array([-70, 0, 0]), [-1, 0, 0]),
]

# ======================================================
# ✅ APLICAR CORTES
# ======================================================
for name, origin, normal in cuts:
    arr = apply_cut(arr, rasToIjk, origin, normal)
    print("✅ Corte aplicado:", name)

# ======================================================
# ✅ Volver a insertar el labelmap y reemplazar segmentos
# ======================================================
flat = arr.ravel(order='F')
vtk_arr = numpy_to_vtk(flat.astype(np.uint16), deep=True)
image2 = vtk.vtkImageData()
image2.DeepCopy(image)
image2.GetPointData().SetScalars(vtk_arr)
labelmapNode.SetAndObserveImageData(image2)

# 🔹 Actualizar segmentación directamente desde el labelmap
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)

# 🔹 Reconstruir superficie
segmentationNode.CreateClosedSurfaceRepresentation()

# 🔹 Refrescar vistas
slicer.util.forceRenderAllViews()

print("🎉 Segmentos reconstruidos y cortes aplicados correctamente.")
