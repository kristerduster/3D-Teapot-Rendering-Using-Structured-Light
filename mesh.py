import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("point_clouds/teapot_merged.ply")

# estimate and orient normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50)) # find up to 50 nearest neighbors within 0.2 units and fit a plane to get normals for poisson
pcd.orient_normals_consistent_tangent_plane(30) # make all normals point the same direction so no flipped mesh faces

# poisson reconstruction/triangle meshing
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=15, width=0, scale=1.1, linear_fit=False
) # scale a little higher than 1 to get bounds

# drop low density/scored triangles that are extrapolated. drop bottom 5%
dens = np.asarray(densities)
keep = dens > np.quantile(dens, 0.03)
mesh = mesh.select_by_index(np.where(keep)[0])

# drop triangles w/ zero area, duplicates, and edges shared by more than 2 triangles
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()

# recompute vertex normals for correct shading and visualization
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh("blender/teapot_merged_poisson.obj", mesh)
print("Saved blender/teapot_merged_poisson.obj")
o3d.visualization.draw_geometries([mesh], window_name="Teapot Poisson", mesh_show_back_face=True)