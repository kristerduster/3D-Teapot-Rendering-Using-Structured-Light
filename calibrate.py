import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import Camera, makerotation, calibratePose, triangulate

# load in the intrinsic camera parameters from 'calibration.pickle'

with open('calibration_C0.pickle', 'rb') as f:
    intrinsics_C0 = pickle.load(f)
fL = (intrinsics_C0['fx'] + intrinsics_C0['fy'])/2 # rn avg out fx fy. if its especially terrible then modify camera class to take fx and fy, modify project fn to apply separate focal lens to each coord
cL = np.array([[intrinsics_C0['cx'], intrinsics_C0['cy']]]).T
distL = intrinsics_C0['dist']

with open('calibration_C1.pickle', 'rb') as f:
    intrinsics_C1 = pickle.load(f)
fR = (intrinsics_C1['fx'] + intrinsics_C1['fy'])/2
cR = np.array([[intrinsics_C1['cx'], intrinsics_C1['cy']]]).T
distR = intrinsics_C1['dist']


# create Camera objects representing the left and right cameras using intrinsics from calibration_*.pickle
camL = Camera(f=fL,c=cL,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))
camR = Camera(f=fR,c=cR,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))

# load in the left and right images and undistort
imgL_raw = plt.imread('images/calib/frame_C0_01.jpg') # 01 seems to be clearest and straightest orientation of the checkerboard idk
KL = np.array([[fL, 0, intrinsics_C0['cx']],
               [0, fL, intrinsics_C0['cy']], # change to fx vs fy here if i end up modifying camera class
               [0, 0, 1]])
imgL = cv2.undistort(imgL_raw, KL, distL)

imgR_raw = plt.imread('images/calib/frame_C1_01.jpg')
KR = np.array([[fR, 0, intrinsics_C1['cx']],
               [0, fR, intrinsics_C1['cy']], 
               [0, 0, 1]])
imgR = cv2.undistort(imgR_raw, KR, distR)


# find coordinates of chessboard corners
ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
pts2L = cornersL.squeeze().T

ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
pts2R = cornersR.squeeze().T

# generate the known 3D point coordinates of points on the checkerboard in cm, 6 rows by 8 columns, square size 2.8 cm
pts3 = np.zeros((3,6*8))
yy,xx = np.meshgrid(np.arange(8),np.arange(6))
pts3[0,:] = 2.8*xx.reshape(1,-1)
pts3[1,:] = 2.8*yy.reshape(1,-1)


# calibrate cams and experiment w initial params 
params_initL = np.array([0,0,0,0,0,-10]) 
camL.update_extrinsics(params_initL)

params_initR = np.array([0,0,0,0,0,-10]) 
camR.update_extrinsics(params_initR)
   
camL = calibratePose(pts3,pts2L,camL,params_initL)
camR = calibratePose(pts3,pts2R,camR,params_initR)

print(camL)
print(camR)

# As a final test, triangulate the corners of the checkerboard to get back their 3D locations
pts3r = triangulate(pts2L, camL, pts2R, camR)

# Display the reprojected points overlayed on the images to make 
# sure they line up
plt.rcParams['figure.figsize']=[15,15]
pts2Lp = camL.project(pts3)
plt.imshow(imgL)
plt.plot(pts2Lp[0,:],pts2Lp[1,:],'bo')
plt.plot(pts2L[0,:],pts2L[1,:],'rx')
plt.show()

pts2Rp = camR.project(pts3)
plt.imshow(imgR)
plt.plot(pts2Rp[0,:],pts2Rp[1,:],'bo')
plt.plot(pts2R[0,:],pts2R[1,:],'rx')
plt.show()

errL = np.linalg.norm(pts2Lp - pts2L, axis=0)
errR = np.linalg.norm(pts2Rp - pts2R, axis=0)
print("Reproj err L: mean %.2f px, max %.2f px" % (errL.mean(), errL.max()))
print("Reproj err R: mean %.2f px, max %.2f px" % (errR.mean(), errR.max()))