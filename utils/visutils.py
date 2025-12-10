import numpy as np
import matplotlib.pyplot as plt

def set_axes_equal_3d(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def label_axes(ax):
    '''Label x,y,z axes with default labels'''
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plot_2d_projections(pts3, colors, bounds):
    """
    Plot 2D projections (XY, XZ, YZ) of colored point cloud.
    
    Parameters
    ----------
    pts3 : 3xN array
        3D points
    colors : Nx3 array
        RGB colors [0, 1]
    bounds : tuple
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x, y, z = pts3[0, :], pts3[1, :], pts3[2, :]
    
    fig = plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, s=1, c=colors)
    plt.xlim([xmin, xmax]); plt.ylim([ymin, ymax])
    plt.xlabel("X"); plt.ylabel("Y"); plt.title("XY Projection")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(x, z, s=1, c=colors)
    plt.xlim([xmin, xmax]); plt.ylim([zmin, zmax])
    plt.xlabel("X"); plt.ylabel("Z"); plt.title("XZ Projection")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(y, z, s=1, c=colors)
    plt.xlim([ymin, ymax]); plt.ylim([zmin, zmax])
    plt.xlabel("Y"); plt.ylabel("Z"); plt.title("YZ Projection")
    plt.grid(True)
    
    plt.tight_layout()
    return fig

def plot_3d_cloud(pts3, colors, cam0, cam1, look0, look1, bounds, title="3D Point Cloud"):
    """
    Plot 3D colored point cloud with camera poses.
    
    Parameters
    ----------
    pts3 : 3xN array
        3D points
    colors : Nx3 array
        RGB colors [0, 1]
    cam0, cam1 : Camera objects
        Left and right camera
    look0, look1 : 3x2 arrays
        Camera look direction lines
    bounds : tuple
        (xmin, xmax, ymin, ymax, zmin, zmax)
    title : str
        Plot title
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x, y, z = pts3[0, :], pts3[1, :], pts3[2, :]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, z, s=1, c=colors, label='Points')
    ax.plot(cam0.t[0], cam0.t[1], cam0.t[2], 'go', markersize=10, label='Left cam')
    ax.plot(cam1.t[0], cam1.t[1], cam1.t[2], 'mo', markersize=10, label='Right cam')
    ax.plot(look0[0,:], look0[1,:], look0[2,:], 'g-', linewidth=2)
    ax.plot(look1[0,:], look1[1,:], look1[2,:], 'm-', linewidth=2)
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    label_axes(ax)
    ax.set_title(title)
    ax.legend(loc='best')
    
    return fig, ax