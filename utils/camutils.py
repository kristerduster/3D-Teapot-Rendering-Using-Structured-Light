import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import pickle
import visutils

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
 