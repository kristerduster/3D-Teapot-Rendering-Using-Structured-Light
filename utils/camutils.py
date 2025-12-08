import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import cv2

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
        
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """

        assert(pts3.shape[0]==3)

        # convert from global to camera coordinate system (apply R^-1(P - t))
        inverse_rotation = np.linalg.inv(self.R)
        converted_pts3 = inverse_rotation @ (pts3 - self.t)
        
        # project into camera (apply /z)
        pts2 = converted_pts3[:2] / converted_pts3[2]
        
        # scale by focal length (apply *f) and offset by principal pt
        pts2 = self.f * pts2 + self.c

        # check the output a little
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
    
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array of shape (6,) (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[:3] are the rotation angles, params[3:] are the translation

        """ 
        self.R = makerotation(*params[:3])
        self.t = params[3:].reshape(3,1)
    
def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    # check size of input
    assert(pts2L.shape[0]==2)
    assert(pts2R.shape[0]==2)

    N = pts2L.shape[1]
    pts3 = np.zeros((3,N), dtype=float)

    # Ax = b
    # A: 3x2 matrix representing (camL.R * (pts2L / camL.f - camL.c/camL.f)) - (camR.R * (pts2R / camR.f - camR.c/camR.f)
    # x: 2x1 matrix [zL zR]
    # b: 3x1 matrix representing camR.t - camL.t
    # solve for x
    for i in range(N): 
        # convert 2d img coordinates into homogeneous coordinates 
        # this already takes into account the principal pt offsets
        qL = np.array([pts2L[0,i], pts2L[1,i], camL.f])
        qR = np.array([pts2R[0,i], pts2R[1,i], camR.f])

        # get principal coordinates of both cameras
        pLx, pLy = camL.c.flatten()
        pRx, pRy = camR.c.flatten()
        
        # compute the (qL/fL - [pLx/fL pLy/fL 0]) ie whatevers left from the RLPL - RRPR = tR-TL after factoring out RL, RR, zL, zR from the left side
        invertedL = (qL/camL.f) - np.array([pLx/camL.f, pLy/camL.f, 0])
        invertedR = (qR/camR.f) - np.array([pRx/camR.f, pRy/camR.f, 0])

        # make A
        A = np.column_stack((camL.R @ invertedL, -camR.R @ invertedR))

        # make b
        b = camR.t - camL.t

        # make x
        x, _, _, _ = np.linalg.lstsq(A,b, rcond=None)
        zL, zR = x

        # reconstruct P using camL
        P = camL.R @ (invertedL * zL) + camL.t.flatten()
        pts3[:,i] = P
    
    # check size of output
    assert(pts3.shape[0]==3)
    
    # and return the reconstructed points
    return pts3

def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """

    # convert to radians
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)

    # construct rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])
    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])
    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    # apply rotation matrices in order: x, y, then z
    R = Rz @ Ry @ Rx

    return R

def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing stored in a vector of shape (6,)

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """
    cam.update_extrinsics(params)
    projected_pts = cam.project(pts3)
    residual = (projected_pts - pts2).flatten()
    return residual

def calibratePose(pts3,pts2,cam,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera
        
    params_init : 1D numpy.array (dtype=float)
        Initial estimate of camera extrinsic parameters ()
        params[0:3] are the rotation angles, params[3:6] are the translation

    Returns
    -------
    cam : Camera
        Refined estimate of camera with updated R,t parameters
        
    """
    # wrap residuals to fix constants (pts3, pts2, cam)
    residual_fn = lambda params: residuals(pts3, pts2, cam, params)

    # optimize
    opt_params, _ = scipy.optimize.leastsq(residual_fn, params_init)
    cam.update_extrinsics(opt_params)

    return cam

def decode(imprefix,start,threshold, K, dist) :
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=int)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    
    nbits = 10
    binary_images = []
    mask_planes = []

    for i in range(nbits):
        filename1 = f"{imprefix}{(start + 2*i):02}.png" # pad with leading zeros/always be 2 digits
        filename2 = f"{imprefix}{(start + 2*i + 1):02}.png" 

        img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.undistort(img1, K, dist)
        img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.undistort(img2, K, dist)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Could not load {filename1} or {filename2}")
        
        img1 = img1.astype(float) / 255.0
        img2 = img2.astype(float) / 255.0

        # img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        # img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

        bit_plane = (img1 > img2).astype(int) # if img1[x,y] > img2[x,y] bit = 1 else 0

        difference = np.abs(img1 - img2)
        mask_plane = (difference >= threshold) # True = above thresh else False

        binary_images.append(bit_plane)
        mask_planes.append(mask_plane)

    bit_stack = np.stack(binary_images, axis=-1) # stack 10 binary imgs into single greycode img
    binary_bits = np.zeros_like(bit_stack) # create array of all 0s same size as bit_stack
    binary_bits[:, :, 0] = bit_stack[:, :, 0] # msb the same

    # convert grey code to binary coded decimal by using xor of successive frames
    for i in range(1, bit_stack.shape[-1]):
        binary_bits[:, :,i] = binary_bits[:, :, i-1] ^ bit_stack[:, :,i] # XOR element wise each binary code bit next to gray code bit in next position

    H, W, _ = binary_bits.shape
    code = np.zeros((H,W), dtype = int)
    
    # convert bcd frames to decimal num for each pixel
    for i in range(nbits):
        code += binary_bits[:, :,i] * (2 ** (nbits - 1 - i))

    # pixel is True iff all 10 planes are True else False
    mask = np.logical_and.reduce(mask_planes)
        
    return code,mask

def reconstruct(imprefixL,imprefixR,threshold,camL,camR, K0, K1, dist0, dist1):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination. This function decodes the binary graycode patterns, matches 
    pixels with corresponding codes, and triangulates the result.
    
    The returned arrays include 2D and 3D coordinates of only those pixels which
    were triangulated where pts3[:,i] is the 3D coordinte produced by triangulating
    pts2L[:,i] and pts2R[:,i]

    Parameters
    ----------
    imprefixL, imprefixR : str
        Image prefixes for the coded images from the left and right camera
        
    threshold : float
        Threshold to determine if a bit is decodeable
   
    camL,camR : Camera
        Calibration info for the left and right cameras
        
    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)
        The 2D pixel coordinates of the matched pixels in the left and right
        image stored in arrays of shape 2xN
        
    pts3 : 2D numpy.array (dtype=float)
        Triangulated 3D coordinates stored in an array of shape 3xN
        
    """

    # Decode the H and V coordinates for the two views
    H_left, Hmask_left = decode(imprefixL, 0, threshold, K0, dist0)
    V_left, Vmask_left = decode(imprefixL, 20, threshold, K0, dist0)

    H_right, Hmask_right = decode(imprefixR, 0, threshold, K1, dist1)
    V_right, Vmask_right = decode(imprefixR, 20, threshold, K1, dist1)

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    code_left = (H_left << 10) + V_left # bit shift H_left by 10
    code_right = (H_right << 10) + V_right

    mask_left = Hmask_left & Vmask_left
    mask_right = Hmask_right & Vmask_right
    
    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    ...
    CL = code_left[mask_left].flatten() # select only pixels where mask is true then flatten into 1d (contains valid pixels' 20 bit code)
    CR = code_right[mask_right].flatten()

    coordsL = np.argwhere(mask_left) # get coordinates of valid pixels 
    coordsR = np.argwhere(mask_right)
    
    common_codes, idxL, idxR = np.intersect1d(CL, CR, return_indices = True) # get indices of pixels with matching common codes
    matchL = coordsL[idxL] # get coordinates of matching pixels
    matchR = coordsR[idxR]

    # Build 2D pixel coordinate arrays from matches
    pts2L = matchL[:, ::-1].T.astype(float)  
    pts2R = matchR[:, ::-1].T.astype(float)
    

    # Now triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)
    
    return pts2L,pts2R,pts3

def project_3D(pts3, K, R, t):
    Pc = R.T @ (pts3 - t)  # 3xN
    z = Pc[2, :]
    uv = K @ (Pc / z)
    u, v = uv[0, :], uv[1, :]
    return u, v, z

