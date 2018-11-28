import pandas as pd
import numpy as np
from data_preprocessing.calculate_euclidean_distance import cal_euclidean_dist
from data_preprocessing.calculate_angle import angle_between

# function to perform exponential weighted moving average and scaling to unit length


def ewma_scale_normalization(dataset, skeletons_dataset, inertial_dataset, jointType, columns, inertial):

    # exponential weighted moving average
    ewma = pd.Series.ewm

    # looping through all the rows in the dataset
    for s in dataset:
        print(s)
        # Joint Smoothing
        sk = skeletons_dataset[s]['sk']
        for j in range(len(sk)):  # looping through the 20 joints
            for c in range(len(sk[j])):  # looping through x, y, z axis
                # take EWMA in both directions with a smaller span term
                coord = pd.Series(sk[j][c])
                fwd = ewma(coord, span=3).mean()  # take EWMA in forward direction
                bwd = ewma(coord[::-1], span=3).mean()  # take EWMA in backward direction
                smoothc = np.vstack((fwd, bwd[::-1]))  # stack fwd and bwd together
                smoothc = np.mean(smoothc, axis=0)  # average of the value in top and bottom row and of same index
                sk[j][c] = smoothc

        # Inertial Smoothing
        inert = inertial_dataset[s]['inertial']

        for v in range(len(inert[0])):  # looping through the 3 x accelerations and 3 x rotations
            # take EWMA in both directions with a smaller span term
            seq = pd.Series(np.array([t[v] for t in inert]))
            fwd = ewma(seq, span=10).mean()  # take EWMA in forward direction
            bwd = ewma(seq[::-1], span=10).mean()  # take EWMA in backward direction
            smoothc = np.vstack((fwd, bwd[::-1]))  # stack fwd and bwd together
            smoothc = np.mean(smoothc, axis=0)  # average of the value in top and bottom row and of same index
            inertial_dataset[s][inertial[v]] = smoothc

        # e.g. y-value of head in initial position minus y-value of hip_center in initial position as the
        height = abs(sk[jointType['head']][1][0] - sk[jointType['hip_center']][1][0])

        # to scale normalization across individuals of different size and height by weighting all distances
        # by the distance between hip-center and head.
        jointVar = {}
        j = 0
        for i in range(len(jointType)):
            for k in range(3):
                jointVar[columns[j]] = sk[i][k] / height
                j = j + 1

        # distances and angles features
        keyVar = {}
        keyVar['hand_d'] = cal_euclidean_dist(sk[jointType['hand_l']], sk[jointType['hand_r']]) / height
        keyVar['foot_d'] = cal_euclidean_dist(sk[jointType['foot_l']], sk[jointType['foot_r']]) / height
        keyVar['hand_foot_l_d'] = cal_euclidean_dist(sk[jointType['hand_l']], sk[jointType['foot_l']]) / height
        keyVar['hand_foot_r_d'] = cal_euclidean_dist(sk[jointType['hand_r']], sk[jointType['foot_r']]) / height
        keyVar['head_hand_l_d'] = cal_euclidean_dist(sk[jointType['head']], sk[jointType['hand_l']]) / height
        keyVar['head_hand_r_d'] = cal_euclidean_dist(sk[jointType['head']], sk[jointType['hand_r']]) / height
        keyVar['elbow_l_a'] = angle_between(sk[jointType['shoulder_l']], sk[jointType['elbow_l']], sk[jointType['wrist_l']])
        keyVar['elbow_r_a'] = angle_between(sk[jointType['shoulder_r']], sk[jointType['elbow_r']], sk[jointType['wrist_r']])
        keyVar['knee_l_a'] = angle_between(sk[jointType['hip_l']], sk[jointType['knee_l']], sk[jointType['foot_l']])
        keyVar['knee_r_a'] = angle_between(sk[jointType['hip_r']], sk[jointType['knee_r']], sk[jointType['foot_r']])
        keyVar['hip_foot_r_d'] = cal_euclidean_dist(sk[jointType['hip_r']], sk[jointType['foot_r']]) / height
        keyVar['hip_foot_l_d'] = cal_euclidean_dist(sk[jointType['hip_l']], sk[jointType['foot_l']]) / height

        for v in jointVar:
            skeletons_dataset[s][v] = jointVar[v]
        for v in keyVar:
            skeletons_dataset[s][v] = keyVar[v]

    return skeletons_dataset, inertial_dataset

