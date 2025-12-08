import matplotlib.pyplot as plt
import numpy as np
import pickle
import open3d as o3d
from utils import decode, reconstruct

fid = open('both_calibrations.pickle','rb')
(camC0, camC1) = pickle.load(fid)
fid.close()

threshold = 0.02

# axis bounds for each grab (visually derived)
bounds = [
    (0, 19.3, 2.8, 22, 18.5, 26.2),      # grab0
    (0, 19.3, 5, 18.3, 18, 27.7),        # grab1
    (0, 19.5, 1.5, 21.7, 18.7, 25.6),    # grab2
    (0, 19.5, 7, 18, 15.5, 24.5),        # grab3
    (0, 19.3, 5, 21.5, 15.6, 24.8),      # grab4
    (7.1, 20, 5, 24, 11.8, 25.6),        # grab5
    (7, 19.7, 8.2, 24, 15, 24.8)         # grab6
]

lookC0 = np.hstack((camC0.t, camC0.t + camC0.R @ np.array([[0,0,100]]).T))  
lookC1 = np.hstack((camC1.t, camC1.t + camC1.R @ np.array([[0,0,100]]).T)) 

for grab_id in range(7):
    print(f"\n=== Processing grab {grab_id} ===")
    
    imprefixC0 = f'images/teapot/grab_{grab_id}_u/frame_C0_'
    imprefixC1 = f'images/teapot/grab_{grab_id}_u/frame_C1_'
    
    pts2L, pts2R, pts3 = reconstruct(imprefixC0, imprefixC1, threshold, camC0, camC1)
    
    # Apply bounds to filter noisy points
    xmin, xmax, ymin, ymax, zmin, zmax = bounds[grab_id]
    mask = (pts3[0,:] >= xmin) & (pts3[0,:] <= xmax) & \
            (pts3[1,:] >= ymin) & (pts3[1,:] <= ymax) & \
            (pts3[2,:] >= zmin) & (pts3[2,:] <= zmax)
    pts3_filtered = pts3[:, mask]
    
    print(f"Before filtering: {pts3.shape[1]} points")
    print(f"After filtering: {pts3_filtered.shape[1]} points")
    
    # Convert to Open3D and denoise
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3_filtered.T)
    pcd_clean, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    
    print(f"After denoising: {len(pcd_clean.points)} points")
    xc, yc, zc = np.asarray(pcd_clean.points).T

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xc, yc, zc, s=1, c='teal', label='Denoised Reconstructed points')
    ax.plot(camC0.t[0], camC0.t[1], camC0.t[2], 'go', markersize=10, label='Left cam')
    ax.plot(camC1.t[0], camC1.t[1], camC1.t[2], 'mo', markersize=10, label='Right cam')
    ax.plot(lookC0[0,:], lookC0[1,:], lookC0[2,:], 'g-', linewidth=2)
    ax.plot(lookC1[0,:], lookC1[1,:], lookC1[2,:], 'm-', linewidth=2)
    ax.set_xlim([xmin,xmax]); ax.set_ylim([ymin,ymax]); ax.set_zlim([zmin,zmax])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Denoised 3D Point Cloud with Cameras - Grab {grab_id}")
    plt.legend(loc='best')
    plt.show()

    # Save to PLY
    output_filename = f'teapot_grab_{grab_id}.ply'
    o3d.io.write_point_cloud(output_filename, pcd_clean)
    print(f"Saved to {output_filename}")

# THRESHOLD EXPERIMENTATION
# for thresh in [0.001, 0.01, 0.02, 0.03, 0.04]:
#     code, mask = decode('images/teapot/grab_6_u/frame_C1_', 20, thresh)
#     plt.figure(figsize=(10,4))
#     plt.suptitle(f"Threshold = {thresh}")
    
#     plt.subplot(1,2,1)
#     plt.imshow(code, cmap='jet')
#     plt.title("Decoded Code")
#     plt.colorbar()
    
#     plt.subplot(1,2,2)
#     plt.imshow(mask, cmap='gray')
#     plt.title("Mask")
#     plt.colorbar()
    
#     plt.show()

# DENOISING EXPERIMENTATION
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts3.T)
# pts3_clean, inliers = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)

# xc, yc, zc = np.asarray(pts3_clean.points).T

# FILTERING EXPERIMENTATION
# print("X range:", pts3[0,:].min(), pts3[0,:].max())
# print("Y range:", pts3[1,:].min(), pts3[1,:].max())
# print("Z range:", pts3[2,:].min(), pts3[2,:].max())

# plot distributions to see where most points lie
# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.hist(pts3[0,:], bins=100); plt.title("X distribution")
# plt.subplot(1,3,2); plt.hist(pts3[1,:], bins=100); plt.title("Y distribution")
# plt.subplot(1,3,3); plt.hist(pts3[2,:], bins=100); plt.title("Z distribution")
# plt.show()

# xmin, xmax, ymin, ymax, zmin, zmax = (0, 19.3, 2.8, 22, 18.5, 26.2) # grab0
# xmin, xmax, ymin, ymax, zmin, zmax = (0, 19.3, 5, 18.3, 18, 27.7) # grab1
# xmin, xmax, ymin, ymax, zmin, zmax = (0, 19.5, 1.5, 21.7, 18.7, 25.6) # grab2
# xmin, xmax, ymin, ymax, zmin, zmax = (0, 19.5, 7, 18, 15.5, 24.5) # grab3
# xmin, xmax, ymin, ymax, zmin, zmax = (0, 19.3, 5, 21.5, 15.6, 24.8) # grab4
# xmin, xmax, ymin, ymax, zmin, zmax = (7.1, 20, 5, 24, 11.8, 25.6) # grab5
# xmin, xmax, ymin, ymax, zmin, zmax = (7, 19.7, 8.2, 24, 15, 24.8) # grab6

# Add your visualization code here.  As we have done previously it is good to visualize different
# 2D projections XY, XZ, YZ and well as a 3D version
# x, y, z = pts3
# lookC0 = np.hstack((camC0.t, camC0.t + camC0.R @ np.array([[0,0,100]]).T))  
# lookC1 = np.hstack((camC1.t, camC1.t + camC1.R @ np.array([[0,0,100]]).T)) 

# plt.figure(figsize=(20,12))

# # XY projection
# plt.subplot(2,3,1)
# plt.scatter(x, y, s=1, c='blue', label='Reconstructed points')
# plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
# plt.xlabel("X"); plt.ylabel("Y"); plt.title("XY Projection")
# plt.legend()
# plt.grid(True)

# # XZ projection
# plt.subplot(2,3,2)
# plt.scatter(x, z, s=1, c='green', label='Reconstructed points')
# plt.xlim([xmin, xmax]); plt.ylim([zmin, zmax])
# plt.xlabel("X"); plt.ylabel("Z"); plt.title("XZ Projection")
# plt.legend()
# plt.grid(True)

# # YZ projection
# plt.subplot(2,3,3)
# plt.scatter(y, z, s=1, c='red', label='Reconstructed points')
# plt.xlim([ymin, ymax]); plt.ylim([zmin, zmax])
# plt.xlabel("Y"); plt.ylabel("Z"); plt.title("YZ Projection")
# plt.legend()
# plt.grid(True)

# # XY projection
# plt.subplot(2,3,4)
# plt.scatter(xc, yc, s=1, c='blue', label='Denoised Reconstructed points')
# plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
# plt.xlabel("X"); plt.ylabel("Y"); plt.title("XY Projection")
# plt.legend()
# plt.grid(True)

# # XZ projection
# plt.subplot(2,3,5)
# plt.scatter(xc, zc, s=1, c='green', label='Denoised Reconstructed points')
# plt.xlim([xmin, xmax]); plt.ylim([zmin, zmax])
# plt.xlabel("X"); plt.ylabel("Z"); plt.title("XZ Projection")
# plt.legend()
# plt.grid(True)

# # YZ projection
# plt.subplot(2,3,6)
# plt.scatter(yc, zc, s=1, c='red', label='Denoised Reconstructed points')
# plt.xlim([ymin, ymax]); plt.ylim([zmin, zmax])
# plt.xlabel("Y"); plt.ylabel("Z"); plt.title("YZ Projection")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# 3D projection
# fig = plt.figure(figsize=(16,8))
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(x, y, z, s=1, c='teal', label='Reconstructed points')
# ax.plot(camC0.t[0], camC0.t[1], camC0.t[2], 'go', markersize=10, label='Left cam')
# ax.plot(camC1.t[0], camC1.t[1], camC1.t[2], 'mo', markersize=10, label='Right cam')
# ax.plot(lookC0[0,:], lookC0[1,:], lookC0[2,:], 'g-', linewidth=2)
# ax.plot(lookC1[0,:], lookC1[1,:], lookC1[2,:], 'm-', linewidth=2)
# ax.set_xlim([xmin,xmax]); ax.set_ylim([ymin,ymax]); ax.set_zlim([zmin,zmax])
# ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
# ax.set_title("3D Point Cloud with Cameras")
# plt.legend(loc='best')

# ax = fig.add_subplot(122, projection='3d')
# ax.scatter(xc, yc, zc, s=1, c='teal', label='Denoised Reconstructed points')
# ax.plot(camC0.t[0], camC0.t[1], camC0.t[2], 'go', markersize=10, label='Left cam')
# ax.plot(camC1.t[0], camC1.t[1], camC1.t[2], 'mo', markersize=10, label='Right cam')
# ax.plot(lookC0[0,:], lookC0[1,:], lookC0[2,:], 'g-', linewidth=2)
# ax.plot(lookC1[0,:], lookC1[1,:], lookC1[2,:], 'm-', linewidth=2)
# ax.set_xlim([xmin,xmax]); ax.set_ylim([ymin,ymax]); ax.set_zlim([zmin,zmax])
# ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
# ax.set_title("Denoised 3D Point Cloud with Cameras")
# plt.legend(loc='best')
# plt.show()