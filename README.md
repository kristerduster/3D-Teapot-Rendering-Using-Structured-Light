# 3D-Teapot-Rendering-Using-Structured-Light

A structured-light 3D reconstruction pipeline for capturing and rendering a physical teapot using two calibrated cameras, gray-code pattern projection, and multi-view point cloud merging.

## Overview

This project reconstructs a 3D model of a teapot from 7 different orientations using structured-light scanning. The workflow consists of five main stages:

1. **Camera Calibration** — Estimate intrinsics and extrinsics for stereo cameras
2. **3D Reconstruction** — Decode structured-light patterns and triangulate points
3. **Point Cloud Merge** — Align and combine 7 individual scans into a single 3D model
4. **Meshing** — Convert point cloud to a triangle mesh using Poisson surface reconstruction
5. **Rendering** — Render the colored mesh in Blender

## Results

**Final Rendering:** [Watch the full 360° teapot rotation](rendered/video/rendering_540fps.mkv)

## Dependencies

```
opencv-python
numpy
scipy
open3d
matplotlib
```

Install via:
```bash
pip install opencv-python numpy scipy open3d matplotlib
```

## Execution Order

```bash
# 1. Calibrate cameras
python calibrate.py

# 2. Reconstruct and color point clouds (7 grabs)
python reconstruct.py

# 3. Merge point clouds (manual alignment + optional ICP)
python merge.py

# 4. Mesh using Poisson
python mesh.py

# 5. Manual hole filling in Blender
# (see Blender workflow below)

# 6. Render and visualize
# (see Rendering section below)
```

## Pipeline

### 1. Camera Calibration (`calibrate.py`)

**Objective:** Estimate camera intrinsics (focal length, principal point, lens distortion) and extrinsics (relative rotation/translation between two cameras).

**Procedure:**
- Generate camera intrinsics (fx, fy, cx, cy, distortion coefficients) for C0 and C1 using OpenCV's `cv2.calibrateCamera()` on 20 pairs of undistorted checkerboard calibration images.
- Calibrate camera extrinsics (R, t) by minimizing reprojection error: undistort checkerboard images, detect corners in both views, and use `calibratePose()` to solve for the transformation between cameras.
- Validate by triangulating detected checkerboard corners and reprojecting them onto original image pairs.

**Key Parameters:**
- Checkerboard inner corners: 8×6
- Square size: 2.8 cm
- Focal length averaging: `f = (fx + fy) / 2` for simplicity (0.01 px difference vs. separate fx/fy)

**Output:**
- `calibration_C0.pickle` and `calibration_C1.pickle` containing intrinsics and distortion coefficients
- `both_calibrations.pickle` containing calibrated Camera objects with extrinsics

**Discussion:**
Undistorting images before calibration ensures consistency with the pinhole camera model. Averaging focal lengths was justified by negligible reconstruction error (< 0.01 px) compared to using separate fx/fy.

---

### 2. 3D Point Cloud Reconstruction (`reconstruct.py`)

**Objective:** Decode structured-light patterns and reconstruct colored 3D point clouds for each of the 7 teapot orientations.

**Procedure:**
- For each grab:
  1. Decode 10-bit gray-code patterns to get per-pixel disparity using `decode()` with threshold = 0.02
  2. Undistort structured-light images to match calibration
  3. Match pixels with identical codes in left/right images and triangulate to 3D using `triangulate()`
  4. Filter points outside per-grab bounding boxes (manually derived from visualization)
  5. Denoise using Open3D statistical outlier removal (nb_neighbors=20, std_ratio=2.5)
  6. Color points by projecting to both cameras and sampling RGB from undistorted foreground images
  7. Build foreground mask via background subtraction (threshold=15) to avoid painting scene background onto teapot
  8. Save colored point cloud as PLY

**Key Parameters:**
- Gray-code threshold: 0.02 (0.01 too fuzzy, >0.02 too many holes)
- Outlier removal: nb_neighbors=20, std_ratio=2.5 (balances density vs. cleanliness)
- Color sampling: prefer C0, fallback to C1 if projection invalid

**Output:**
- `teapot_grab_0.ply` through `teapot_grab_6.ply` — colored point clouds (3D points + vertex RGB)

**Discussion:**
Undistorting all images before processing ensures geometric consistency with calibrated cameras. Higher `nb_neighbors` (e.g., 30) removes more outliers but creates holes; lower `std_ratio` is more aggressive. Glare on shiny teapot surfaces created holes (esp. grab 0), but merging multiple views mitigates this. Color projection to both cameras provides robustness to occlusions and variable lighting per camera.

---

### 3. Point Cloud Merge (`merge.py`)

**Objective:** Align 7 individual point clouds into a single global coordinate system using manual alignment + ICP refinement.

**Procedure:**
- Center each point cloud at its geometric center in code
- Import centered PLYs to Blender and manually rotate/translate each to align with grab 0
- Record Location (x, y, z) and Rotation (x, y, z in degrees) for each grab from Blender's transform panel
- Convert Blender transforms (in meters) to centimeters and apply as 4×4 matrices
- Run ICP on top for sub-millimeter refinement (threshold=0.5 cm, max_iterations=50)
- Merge aligned clouds and downsample (voxel_size=0.1 cm)
- Center final merged cloud at origin

**Key Parameters:**
- Blender location unit: **meters** (must multiply by 100 to convert to cm)
- Blender rotation: **degrees** (converted to radians for scipy Rotation)
- ICP threshold: 0.5 cm (max correspondence distance)

**Output:**
- `teapot_merged.ply` — merged, colored point cloud centered at origin

**Discussion:**
Automatic ICP alone failed to align grabs properly, likely due to non-uniform turntable rotation or insufficient point overlap. Manual alignment in Blender provided visual feedback and coarse registration; ICP refinement was then attempted but provided minimal improvement and sometimes made alignment worse. Final alignment achieved through manual Blender adjustment only (no ICP). Manual alignment was extremely time-consuming due to tiny visual features and required multiple re-starts (losing transformation data after accidentally using "Set Origin to Geometry").

---

### 4. Meshing (`mesh.py`)

**Objective:** Convert point cloud to a triangle mesh.

**Procedure:**
- Load merged colored point cloud
- Estimate vertex normals (radius=0.2, max_nn=50)
- Orient normals consistently using `orient_normals_consistent_tangent_plane()`
- Run Poisson surface reconstruction (depth=9, scale=1.1)
- Trim low-density vertices (bottom 5% of density, indicating sparse/extrapolated regions)
- Clean mesh: remove degenerate, duplicate, and non-manifold triangles
- Recompute vertex normals for smooth shading

**Alternatives Tested:**
- **Delaunay triangulation:** Convex hull-like output with holes; edge-length pruning created inconsistent results (0.02 cm → holes; 8 cm → jewel-like artifact)
- **Ball Pivoting:** More triangles, better concavity capture than Delaunay, but jagged appearance
- **Poisson (chosen):** Smoothest surface, fewest holes, best overall quality

**Output:**
- `teapot_merged_poisson.ply` — triangle mesh with vertex colors

**Discussion:**
Poisson reconstruction leverages normal orientation to solve an implicit surface equation, producing smooth, watertight output. Density-based trimming removes spurious floating geometry while preserving the main surface. The resulting mesh still had significant holes (esp. handle/spout), requiring manual filling in Blender.

---

### 5. Manual Hole Filling (Blender)

**Procedure:**
1. Import `teapot_merged_poisson.ply` into Blender
2. Enter Edit Mode (Tab)
3. Switch to Wireframe view (Z → Wireframe) for visibility
4. Zoom in close to holes (adjust Clip Start to 0.001 if needed)
5. Select boundary edges via `Alt+Click` (edge loop selection) or `Select → By Trait → Non Manifold`
6. Fill holes: Press `F` (Face) or `Mesh → Face → Fill`
7. Smooth filled areas if needed (Laplacian Smooth)
8. Exit Edit Mode and export as PLY with vertex colors enabled

**Output:**
- `teapot_filled.ply` (or similar) — mesh with manually filled holes

**Discussion:**
Automatic hole-filling was avoided due to risk of accidentally bridging the handle to the body. Larger holes (spout, handle underside) were difficult to fill perfectly, resulting in slightly geometric/faceted appearance in those regions.

---

### 6. Rendering & Visualization (Blender)

**Procedure:**
1. Import hole-filled colored mesh into Blender
2. Add Shading material:
   - Attribute node (name: `Col`) → Principled BSDF Base Color
3. Add lights (e.g., Area light for diffuse illumination)
4. Adjust roughness and other material properties for desired look
5. Rotate teapot 360° around each axis (X, Y, Z) using keyframe animation
6. Render in Cycles engine with appropriate exposure/gamma
7. Collect frames into video

**Output:**
- Rendered image sequences showing teapot from all angles
- Merged video showing full 360° rotation

**Discussion:**
Vertex colors from the point cloud sampling are preserved in the PLY and automatically used in Blender via the Attribute node. Initial renders were too dark; adding area lights and adjusting exposure improved visibility. Camera path animation was replaced with teapot rotation for simplicity.

---

## Project Structure

```
.
├── calibrate.py              # Camera intrinsic/extrinsic calibration
├── reconstruct.py            # Decode structured light & color point clouds (7 grabs)
├── merge.py                  # Align & merge point clouds via manual + ICP
├── mesh.py                   # Poisson surface reconstruction
├── utils/
│   ├── __init__.py
│   ├── calibutils.py         # Intrinsics loading helper
│   ├── camutils.py           # Camera projection, triangulation, pose estimation
│   ├── colorutils.py         # Color projection & foreground masking
│   └── visutils.py           # Visualization functions (2D/3D plotting)
├── images/
│   ├── calib/                # 20 pairs of stereo checkerboard calibration images
│   └── teapot/
│       ├── grab_0_u/ ... /grab_6_u/  # 7 turntable orientations (structured light + color)
├── calibration_pickles/      # Saved intrinsics/extrinsics
├── point_clouds/             # Intermediate PLYs (individual grabs, merged)
├── blender/                  # Blender project files (alignment, hole filling, rendering)
├── rendered/                 # Final rendered images/video
└── README.md
```

## Key Files

| File | Purpose |
|------|---------|
| `calibrate.py` | Estimate camera intrinsics/extrinsics from checkerboard images |
| `reconstruct.py` | Decode structured light, colorize, and save 7 point clouds |
| `merge.py` | Align & merge using manual transforms + optional ICP |
| `mesh.py` | Poisson surface reconstruction + density-based trimming |
| `utils/calibutils.py` | Load intrinsics from pickle files |
| `utils/camutils.py` | Camera model, projection, triangulation (modified for separate fx/fy) |
| `utils/colorutils.py` | Color projection, background subtraction, foreground masking |
| `utils/visutils.py` | 2D/3D visualization helpers |