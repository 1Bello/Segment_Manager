import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# --------------------------
# Parámetros
# --------------------------
case_name = "CASO_CUELLO_5"
segmentName = "trachea"

segmentationNode = slicer.util.getNode(case_name)
volumeNode = slicer.util.getNode(case_name)

print(f"✅ Nodos Encontrados — Segmentación: {segmentationNode.GetName()}, Volumen: {volumeNode.GetName()}")

# --------------------------
# Calcular centro de la tráquea
# --------------------------
def get_segment_center(segmentationNode, segmentName):
    segmentation = segmentationNode.GetSegmentation()
    segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, [segmentId], labelmapNode)

    image = labelmapNode.GetImageData()
    dims = image.GetDimensions()
    arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1])
    coords = np.argwhere(arr > 0)
    ijk_center = coords.mean(axis=0)[::-1]

    ijk_to_ras = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijk_to_ras)
    ras = [0, 0, 0, 1]
    ijk_to_ras.MultiplyPoint(list(ijk_center) + [1], ras)
    slicer.mrmlScene.RemoveNode(labelmapNode)
    return ras[:3]

center = get_segment_center(segmentationNode, segmentName)
print("📌 Trachea center (RAS):", center)

import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# --------------------------
# Parámetros iniciales
# --------------------------
case_name = "CASO_CUELLO_5"
segmentName = "trachea"

segmentationNode = slicer.util.getNode(case_name)
volumeNode = slicer.util.getNode(case_name)

print(f"✅ Nodos encontrados — Segmentación: {segmentationNode.GetName()}, Volumen: {volumeNode.GetName()}")

# --------------------------
# Función: Obtener centro del segmento
# --------------------------
def get_segment_center(segmentationNode, segmentName):
    segmentation = segmentationNode.GetSegmentation()
    segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, [segmentId], labelmapNode)

    image = labelmapNode.GetImageData()
    dims = image.GetDimensions()
    arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1])
    coords = np.argwhere(arr > 0)
    ijk_center = coords.mean(axis=0)[::-1]

    ijk_to_ras = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijk_to_ras)
    ras = [0, 0, 0, 1]
    ijk_to_ras.MultiplyPoint(list(ijk_center) + [1], ras)
    slicer.mrmlScene.RemoveNode(labelmapNode)
    return ras[:3]

# --------------------------
# Función: Aplicar corte
# --------------------------
def apply_cut(segmentationNode, origin_ras, normal_ras, label="CutPlane"):
    import vtk, numpy as np
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    # Exportar todos los segmentos a LabelMap
    labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapNode)
    image = labelmapNode.GetImageData()
    
    rasToIjk = vtk.vtkMatrix4x4()
    labelmapNode.GetRASToIJKMatrix(rasToIjk)

    # Convertir origen y normal a IJK
    origin_ras4 = list(origin_ras) + [1]
    normal_ras4 = list(normal_ras) + [0]
    planeOrigin_ijk = np.array(rasToIjk.MultiplyDoublePoint(origin_ras4)[:3])
    planeNormal_ijk = np.array(rasToIjk.MultiplyDoublePoint(normal_ras4)[:3])
    planeNormal_ijk /= np.linalg.norm(planeNormal_ijk)

    # Visualizar plano
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetCenter(*origin_ras)
    planeSource.SetNormal(*normal_ras)
    planeSource.Update()

    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", label)
    modelNode.SetAndObservePolyData(planeSource.GetOutput())
    modelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
    modelDisplay.SetColor(1, 0, 0)
    modelDisplay.SetOpacity(0.3)
    slicer.mrmlScene.AddNode(modelDisplay)
    modelNode.SetAndObserveDisplayNodeID(modelDisplay.GetID())

    # Aplicar corte voxel a voxel
    dims = image.GetDimensions()
    arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1])
    zz, yy, xx = np.meshgrid(np.arange(dims[2]), np.arange(dims[1]), np.arange(dims[0]), indexing='ij')
    coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)

    diff = coords - planeOrigin_ijk
    mask = np.sum(diff * planeNormal_ijk, axis=-1) > 0
    arr[mask] = 0

    # Guardar imagen modificada
    flat = arr.reshape(-1)
    vtk_arr = numpy_to_vtk(flat.astype(np.uint16), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    image_mod = vtk.vtkImageData()
    image_mod.DeepCopy(image)
    image_mod.GetPointData().SetScalars(vtk_arr)
    labelmapNode.SetAndObserveImageData(image_mod)

    # Reimportar en segmentación
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
    slicer.mrmlScene.RemoveNode(labelmapNode)
    segmentationNode.CreateClosedSurfaceRepresentation()

    print(f"✅ Corte aplicado: {label}")

# --------------------------
# Calcular centro de tráquea y límites
# --------------------------
center = get_segment_center(segmentationNode, segmentName)
print("📌 Trachea center (RAS):", center)

bounds = [0]*6
volumeNode.GetRASBounds(bounds)
print(f"📏 Bounds — X:[{bounds[0]}, {bounds[1]}] Y:[{bounds[2]}, {bounds[3]}] Z:[{bounds[4]}, {bounds[5]}]")

# --------------------------
# Aplicar cortes múltiples
# --------------------------

# 1️⃣ Espalda (posterior)
posterior_limit = bounds[3]
offset_back = 110
plane_y = (center[1] + posterior_limit) / 2.0 - offset_back
apply_cut(segmentationNode, [center[0], plane_y, center[2]], [0, -1, 0], label="Cut_Back")

# 2️⃣ Cabeza (superior)
offset_head = 50
plane_z = bounds[5] - offset_head
apply_cut(segmentationNode, [center[0], center[1], plane_z], [0, 0, -1], label="Cut_Head")

# 3️⃣ Hombro derecho
offset_shoulder = 60
plane_x_right = bounds[1] - offset_shoulder
apply_cut(segmentationNode, [plane_x_right, center[1], center[2]], [-1, 0, 0], label="Cut_RightShoulder")

# 4️⃣ Hombro izquierdo
plane_x_left = bounds[0] + offset_shoulder
apply_cut(segmentationNode, [plane_x_left, center[1], center[2]], [1, 0, 0], label="Cut_LeftShoulder")

print("✅ Todos los cortes aplicados y visibles.")


# --------------------------
# Exportar a LabelMap
# --------------------------
labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapNode)
image = labelmapNode.GetImageData()

rasToIjk = vtk.vtkMatrix4x4()
labelmapNode.GetRASToIJKMatrix(rasToIjk)

# Convertir plano RAS→IJK
origin_ras = np.array([center[0], plane_y, center[2], 1])
planeOrigin_ijk = rasToIjk.MultiplyDoublePoint(origin_ras)
planeOrigin_ijk = np.array(planeOrigin_ijk[:3])

normal_ras = np.array([0, -1, 0, 0])  # hacia anterior
planeNormal_ijk = rasToIjk.MultiplyDoublePoint(normal_ras)
planeNormal_ijk = np.array(planeNormal_ijk[:3])
planeNormal_ijk /= np.linalg.norm(planeNormal_ijk)

planeSource = vtk.vtkPlaneSource()
planeSource.SetCenter(center[0], plane_y, center[2])
planeSource.SetNormal(0, -1, 0)
planeSource.Update()

modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CutPlaneViz")
modelNode.SetAndObservePolyData(planeSource.GetOutput())
modelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
modelDisplay.SetColor(1, 0, 0)
modelDisplay.SetOpacity(0.3)
slicer.mrmlScene.AddNode(modelDisplay)
modelNode.SetAndObserveDisplayNodeID(modelDisplay.GetID())

# --------------------------
# Aplicar corte voxel a voxel
# --------------------------
dims = image.GetDimensions()
arr = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1])

# Coordenadas ijk por voxel
zz, yy, xx = np.meshgrid(np.arange(dims[2]), np.arange(dims[1]), np.arange(dims[0]), indexing='ij')
coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)

# Producto punto para saber qué voxeles quedan detrás del plano
diff = coords - planeOrigin_ijk
mask = np.sum(diff * planeNormal_ijk, axis=-1) > 0  # voxeles posteriores
arr[mask] = 0  # eliminar parte posterior

# --------------------------
# Guardar nueva imagen
# --------------------------
flat = arr.reshape(-1)
vtk_arr = numpy_to_vtk(flat.astype(np.uint16), deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
image_mod = vtk.vtkImageData()
image_mod.DeepCopy(image)
image_mod.GetPointData().SetScalars(vtk_arr)

labelmapNode.SetAndObserveImageData(image_mod)

# --------------------------
# Reimportar en segmentación
# --------------------------
slicer.mrmlScene.RemoveNode(labelmapNode)
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)
segmentationNode.CreateClosedSurfaceRepresentation()


slicer.util.forceRenderAllViews()
print("✅ Corte aplicado y persistente sobre todos los segmentos.3")
