import numpy as np


def sample_cylinder_at_pose(pose_H):
    """
        Sample cylinder at given pose

        args:
            pose_H: (4x4 np array) Homogeneous transform from inertial frame to cylinder "c.o.g."
        
        returns:
            rotated_samples: (list) List of x,y,z points on the exterior of the cylinder at centered at pose_H
        
    """

# # fake GT pose
#     pose = np.eye(4,4)
#     # rot pi/2 about x
#     pose[0:3, 0:3] = np.array([  [0.0000000, -1.0000000,  0.0000000],
#    [0.0000000,  0.0000000, -1.0000000],
#    [1.0000000,  0.0000000,  0.0000000] ])
    # grab rot and origin
    rot = pose_H[0:3,0:3] 
    trans = pose_H[0:3,3]
    radius = 67 # mm
    height = 100 # mm
    samples = []
    theta = 0
    z_inc = 0.5
    z = -height/2
    # sample points in cylinder frame
    while z < height/2:
        while theta < 2 * np.pi:
            x = trans[0] + radius * np.cos(theta)
            y = trans[1] + radius * np.sin(theta)
            z = trans[2] + z
            point = np.array([x, y, z])
            samples.append(point)
            theta += np.pi/6
        theta = 0
        z+=z_inc
    # apply transformation
    rotated_samples = []
    for sample in samples:
        rotated_sample = np.dot(rot, sample)
        rotated_samples.append(rotated_sample)
    return rotated_samples
