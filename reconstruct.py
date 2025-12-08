import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import decode, reconstruct


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

# thresh0C0V = 0.02
# thresh0C0H = 0.02
# thresh0C1V = 0.02
# thresh0C1H = 0.02

# thresh1C0V = 0.02
# thresh1C0H = 0.02
# thresh1C1V = 0.02
# thresh1C1H = 0.02

# thresh2C0V = 0.02
# thresh2C0H = 0.02
# thresh2C1V = 0.02
# thresh2C1H = 0.02

# thresh3C0V = 0.02
# thresh3C0H = 0.02
# thresh3C1V = 0.02
# thresh3C1H = 0.02

# thresh4C0V = 0.02
# thresh4C0H = 0.02
# thresh4C1V = 0.02
# thresh4C1H = 0.02

# thresh5C0V = 0.02
# thresh5C0H = 0.02
# thresh5C1V = 0.02
# thresh5C1H = 0.02

# thresh6C0V = 0.02
# thresh6C0H = 0.02
# thresh6C1V = 0.02
# thresh6C1H = 0.02


#
# Reconstruct and visualize the results
#
imprefixC0 = 'images/teapot/grab_0_u/frame_C0_'
imprefixC1 = 'images/teapot/grab_0_u/frame_C1_'
threshold = 0.02

fid = open('both_calibrations.pickle','rb')
(camC0,camC1) = pickle.load(fid)
fid.close

print(camC0)
print(camC1)

pts2L,pts2R,pts3 = reconstruct(imprefixC0,imprefixC1,threshold,camC0,camC1)
print("X range:", pts3[0,:].min(), pts3[0,:].max())
print("Y range:", pts3[1,:].min(), pts3[1,:].max())
print("Z range:", pts3[2,:].min(), pts3[2,:].max())

# # plot distributions to see where most points lie
# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.hist(pts3[0,:], bins=100); plt.title("X distribution")
# plt.subplot(1,3,2); plt.hist(pts3[1,:], bins=100); plt.title("Y distribution")
# plt.subplot(1,3,3); plt.hist(pts3[2,:], bins=100); plt.title("Z distribution")
# plt.show()

xmin, xmax, ymin, ymax, zmin, zmax = (0, 18.5, 2.8, 22, 18.5, 26.2)

# Add your visualization code here.  As we have done previously it is good to visualize different
# 2D projections XY, XZ, YZ and well as a 3D version
x, y, z = pts3
lookC0 = np.hstack((camC0.t, camC0.t + camC0.R @ np.array([[0,0,100]]).T))  
lookC1 = np.hstack((camC1.t, camC1.t + camC1.R @ np.array([[0,0,100]]).T)) 

plt.figure(figsize=(15,5))

# XY projection
plt.subplot(1,3,1)
plt.scatter(x, y, s=1, c='blue', label='Reconstructed points')
plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
plt.xlabel("X"); plt.ylabel("Y"); plt.title("XY Projection")
plt.legend()
plt.grid(True)

# XZ projection
plt.subplot(1,3,2)
plt.scatter(x, z, s=1, c='green', label='Reconstructed points')
plt.xlim([xmin, xmax]); plt.ylim([zmin, zmax])
plt.xlabel("X"); plt.ylabel("Z"); plt.title("XZ Projection")
plt.legend()
plt.grid(True)

# YZ projection
plt.subplot(1,3,3)
plt.scatter(y, z, s=1, c='red', label='Reconstructed points')
plt.xlim([ymin, ymax]); plt.ylim([zmin, zmax])
plt.xlabel("Y"); plt.ylabel("Z"); plt.title("YZ Projection")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D projection
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, c='teal', label='Reconstructed points')
ax.plot(camC0.t[0], camC0.t[1], camC0.t[2], 'go', markersize=10, label='Left cam')
ax.plot(camC1.t[0], camC1.t[1], camC1.t[2], 'mo', markersize=10, label='Right cam')
ax.plot(lookC0[0,:], lookC0[1,:], lookC0[2,:], 'g-', linewidth=2)
ax.plot(lookC1[0,:], lookC1[1,:], lookC1[2,:], 'm-', linewidth=2)
ax.set_xlim([xmin,xmax]); ax.set_ylim([ymin,ymax]); ax.set_zlim([zmin,zmax])
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("3D Point Cloud with Cameras")
plt.legend(loc='best')
plt.show()