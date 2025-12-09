import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# center point clouds about the origin
for grab_id in range(7):
    pcd = o3d.io.read_point_cloud(f'teapot_grab_{grab_id}.ply')
    center = pcd.get_center()
    print(f"Grab {grab_id}: center = {center}")
    
    # Translate to origin
    pcd.translate(-center)
    
    # Save centered
    o3d.io.write_point_cloud(f'teapot_grab_{grab_id}.ply', pcd)
    print(f"  Saved centered to teapot_grab_{grab_id}.ply")


blender_transforms = {
    0: {"loc": [4.2705, -0.54219, 9.9963], "rot": [-55.073, 93.835, -60.89]}, 
    1: {"loc": [2.3236, 3.8377, 9.2822], "rot": [-54.731, 93.626, 355.09]},  
    2: {"loc": [-4.0734, -0.76932, 9.7525], "rot": [151.16, 92.095, -31.85]},
    3: {"loc": [0.84905, -6.1091, 9.8981], "rot": [-1.8542, 94.933, -86.752]},
    4: {"loc": [3.8216, 2.4881, 9.4558], "rot": [-43.44, 92.519, -19.578]},
    5: {"loc": [-1.5232, -0.91619, 15.809], "rot": [-3.4798, -16.161, 27.592]},
    6: {"loc": [0.000071, 0.003916, 0.0061033], "rot": [-176.74, -13.126, 151.84]},
}

def blender_to_matrix(loc, rot_deg):
    """Convert Blender location & rotation to 4x4 transformation matrix"""
    # Rotation in degrees → radians
    rot_rad = np.radians(rot_deg)
    # Create rotation matrix (Blender uses XYZ Euler angles)
    r = Rotation.from_euler('xyz', rot_rad)
    
    # Build 4x4 matrix
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = loc
    return mat


print("=== Merging 7 teapot point clouds ===\n")

aligned_clouds = []

# perform manual alignment
for grab_id in range(0, 7):
    print(f"\nAligning grab {grab_id} to reference...")
    source = o3d.io.read_point_cloud(f'teapot_grab_{grab_id}.ply')
    print(f"  Loaded: {len(source.points)} points")

    # Get Blender transform
    loc = blender_transforms[grab_id]["loc"]
    rot = blender_transforms[grab_id]["rot"]
    trans_matrix = blender_to_matrix(loc, rot)

    # Apply transformation
    source.transform(trans_matrix)
    aligned_clouds.append(source)

# Merge all aligned clouds
print("\nMerging aligned clouds...")
merged = o3d.geometry.PointCloud()
for cloud in aligned_clouds:
    merged += cloud

# Save merged cloud
o3d.io.write_point_cloud('teapot_merged.ply', merged)
print("\nSaved: teapot_merged.ply")

# Visualize result
print("Visualizing merged cloud...")
o3d.visualization.draw_geometries([merged], window_name="Merged Teapot", width=1024, height=768)