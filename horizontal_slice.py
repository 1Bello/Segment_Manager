# ======================================================
# ✅ Crear capas horizontales (separación por Z)
# ======================================================

# Espesor de capa en número de voxeles (ajústalo)
layer_thickness = 15  # por ejemplo 20 voxeles de altura

X, Y, Z = arr.shape
num_layers = Z // layer_thickness + 1

print(f"🔹 Capas horizontales: {num_layers} (espesor: {layer_thickness} voxeles)")

layers = []

for i in range(num_layers):
    z_start = i * layer_thickness
    z_end = min((i+1) * layer_thickness, Z)

    print(f"➡️  Generando capa {i}: Z = {z_start} → {z_end}")

    # Crear una máscara solo con la capa actual
    layer_arr = np.zeros_like(arr)
    layer_arr[:, :, z_start:z_end] = arr[:, :, z_start:z_end]

    # Convertir a vtkImageData
    flat = layer_arr.ravel(order='F')
    vtk_arr = numpy_to_vtk(flat.astype(np.uint16), deep=True)

    imageLayer = vtk.vtkImageData()
    imageLayer.DeepCopy(image)
    imageLayer.GetPointData().SetScalars(vtk_arr)

    # Crear LabelMap para la capa
    layerNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        f"Layer_{i}"
    )
    layerNode.SetAndObserveImageData(imageLayer)

    # Crear Segmentación para la capa
    segNodeLayer = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode",
        f"Seg_Layer_{i}"
    )

    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        layerNode, segNodeLayer
    )

    segNodeLayer.CreateClosedSurfaceRepresentation()

    layers.append(segNodeLayer)

print("🎉 Todas las capas horizontales creadas correctamente.")
