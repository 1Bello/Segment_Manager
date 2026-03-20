import os
import slicer
import vtk

# exec(open(r'd:/3D slicer code/Impresion-Medicina/load_case.py').read())
# -----------------------------
# Configuration
# -----------------------------
case_id = 6
case_name = f"CASO_CUELLO_{case_id}"
base_dir = r"D:\3D slicer code\Impresion-Medicina\Cuello"
image_path = os.path.join(base_dir, "imagenes", f"{case_name}.nii.gz")
seg_dir = os.path.join(base_dir, "segmentations", f"{case_name}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")
if not os.path.isdir(seg_dir):
    raise FileNotFoundError(f"Segmentation folder not found: {seg_dir}")

# 1. Load base image
print(f"Loading volume: {image_path}")
volumeNode = slicer.util.loadVolume(image_path)
if volumeNode is None:
    raise RuntimeError(f"Failed to load volume: {image_path}")

# 2. Create empty segmentation container
# (AddNewNodeByClass already adds it to the scene)
segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLSegmentationNode", case_name
)
if segmentationNode is None:
    raise RuntimeError("Could not create segmentation node")
segmentationNode.CreateDefaultDisplayNodes()  # needed for 3D view
print(f"Created segmentation node: {segmentationNode.GetID()}")

# 3. Load each binary mask (one .nii per structure) and import
logic = slicer.modules.segmentations.logic()
added_segments = []

for file in sorted(os.listdir(seg_dir)):

    if not file.lower().endswith(".nii.gz"):
        continue

    
    #if file.lower() in [ "body_trunc.nii", "body_trunc.nii.gz", "eye_left.nii", "eye_left.nii.gz", "eye_lens_left.nii", "eye_lens_left.nii.gz", "eye_lens_right.nii", "eye_lens_right.nii.gz", "eye_right.nii", "eye_right.nii.gz", "body.nii", "body.nii.gz", "body_extremities.nii", "body_extremities.nii.gz", "hard_palate.nii", "hard_palate.nii.gz", "torso_fat.nii", "torso_fat.nii.gz", "head.nii", "head.nii.gz"]:
    #    print(f"Skipping segment: {file}")
    #    continue
    
    # 🌟 MODIFICACIÓN CLAVE: Limpiar el nombre del segmento 🌟
    # Esto asegura que "trachea.nii.gz" se convierta en "trachea"
    labelName = file
    if labelName.lower().endswith(".nii.gz"):
        labelName = labelName[:-7]  # Quita .nii.gz
    elif labelName.lower().endswith(".nii"):
        labelName = labelName[:-4]  # Quita .nii

    seg_path = os.path.join(seg_dir, file)
    print(f"Importing labelmap: {file} -> segment '{labelName}'")

    labelNode = slicer.util.loadLabelVolume(seg_path)
    if not labelNode:
        print(f"  [WARN] Could not load label volume: {seg_path}")
        continue

    # Import the labelmap as segment(s)
    logic.ImportLabelmapToSegmentationNode(labelNode, segmentationNode)

    # Retrieve the last added segment ID (assumes one segment per file)
    segmentIds = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)
    if segmentIds.GetNumberOfValues() == 0:
        print(f"  [WARN] No segments present after import of {file}")
        continue
    newSegmentId = segmentIds.GetValue(segmentIds.GetNumberOfValues() - 1)
    segment = segmentationNode.GetSegmentation().GetSegment(newSegmentId)
    if segment is None:
        print(f"  [WARN] Could not access newly added segment for {file}")
        continue
    segment.SetName(labelName) # Usa el nombre limpio
    added_segments.append(labelName)

    # Optional cleanup to keep scene tidy
    slicer.mrmlScene.RemoveNode(labelNode)

print(f"Imported {len(added_segments)} segments: {', '.join(added_segments)}")

# -----------------------------
# 3.5 Apply smoothing + morphological opening to clean small voxels
# -----------------------------
segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setSourceVolumeNode(volumeNode)  # actualizado a Slicer moderno

# ... (código de smoothing y islands comentado, se deja igual)

# 4. Ensure closed surface representation exists for 3D display
try:
    segmentationNode.CreateClosedSurfaceRepresentation()
except AttributeError:
    # Older Slicer versions: explicitly request representation via segmentation object
    closedSurfaceName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
    if not segmentationNode.GetSegmentation().ContainsRepresentation(
        closedSurfaceName
    ):
        segmentationNode.GetSegmentation().CreateRepresentation(
            closedSurfaceName
        )

# 5. Enable visualization settings
displayNode = segmentationNode.GetDisplayNode()
if displayNode:
    displayNode.SetVisibility3D(True)
    displayNode.SetVisibility2DFill(True)
    displayNode.SetVisibility2DOutline(True)

print(f"Loaded case {case_id} with all segmentations.")

# 6. Center / reset 3D view (and slice views) to show the loaded data
try:
    lm = slicer.app.layoutManager()
    # Reset 3D cameras
    for threeDViewIndex in range(lm.threeDViewCount):
        threeDView = lm.threeDWidget(threeDViewIndex).threeDView()
        threeDView.resetFocalPoint()
        threeDView.resetCamera()
    # Fit slice views to volume
    slicer.app.applicationLogic().FitSliceToAll()
    print("3D and slice views centered on loaded data.")
except Exception as e:
    print(f"[WARN] Could not reset views: {e}")