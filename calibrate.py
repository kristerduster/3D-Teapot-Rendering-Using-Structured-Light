import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import Camera, makerotation, calibratePose, triangulate

# load in the intrinsic camera parameters from 'calibration.pickle'
...
with open('calibration.pickle', 'rb') as f:
    intrinsics = pickle.load(f)
# print(intrinsics)
f = (intrinsics['fx'] + intrinsics['fy'])/2
c = np.array([[intrinsics['cx'], intrinsics['cy']]]).T
# print(f)

# create Camera objects representing the left and right cameras
# use the known intrinsic parameters you loaded in.
camL = Camera(f=f,c=c,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))
camR = Camera(f=f,c=c,t=np.array([[0,0,0]]).T, R=makerotation(0,0,0))

# load in the left and right images and find the coordinates of
# the chessboard corners using OpenCV
imgL = plt.imread('calib1/calib1/Left.jpg')
ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
pts2L = cornersL.squeeze().T

imgR = plt.imread('calib1/calib1/Right.jpg')
ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
pts2R = cornersR.squeeze().T

# generate the known 3D point coordinates of points on the checkerboard in cm
pts3 = np.zeros((3,6*8))
yy,xx = np.meshgrid(np.arange(8),np.arange(6))
pts3[0,:] = 2.8*xx.reshape(1,-1)
pts3[1,:] = 2.8*yy.reshape(1,-1)


# Now use your calibratePose function to get the extrinsic parameters
# for the two images. You may need to experiment with the initialization
# in order to get a good result

...

params_initL = np.array([0,0,0,0,0,-10]) 
camL.update_extrinsics(params_initL)

params_initR = np.array([0,0,0,0,0,-10]) 
camR.update_extrinsics(params_initR)
   
camL = calibratePose(pts3,pts2L,camL,params_initL)
camR = calibratePose(pts3,pts2R,camR,params_initR)

print(camL)
print(camR)

# As a final test, triangulate the corners of the checkerboard to get back there 3D locations
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