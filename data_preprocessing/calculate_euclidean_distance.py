import numpy as np


def cal_euclidean_dist(a, b):
    # Input: ndarrays of joint a and joint b
    ax, bx, ay, by, az, bz = a[0], b[0], a[1], b[1], a[2], b[2]
    output = []
    for i in range(len(ax)):
        output.append(np.linalg.norm(np.array([ax[i], ay[i], az[i]]) - np.array([bx[i], by[i], bz[i]])))
    return np.array(output)