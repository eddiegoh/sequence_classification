import numpy as np


def angle_between(a, b, c):
    ax, bx, cx, ay, by, cy, az, bz, cz = a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]
    angles = []
    for i in range(len(ax)):
        au = np.array([ax[i],ay[i],az[i]])
        bu = np.array([bx[i],by[i],bz[i]])
        cu = np.array([cx[i],cy[i],cz[i]])
        du = au - bu
        angle = np.arccos(np.clip(np.dot(du, cu), -1, 1)) / (np.linalg.norm(du) * np.linalg.norm(cu))
        if angle > (np.pi/2):
            angles.append((2*np.pi-angle)/np.pi)
        else:
            angles.append(angle/np.pi)
    return np.array(angles)