import pandas as pd
import pickle as pk
from data_preprocessing.data_loading import data_load
from data_preprocessing.frame_rescaling import rescale
from data_preprocessing.smoothing_normalization import ewma_scale_normalization
from data_preprocessing.normalization import normalise


# load the list of file names inertial and skeleton data
dataset, inertial_dataset, skeletons_dataset = data_load()

# 20 joints in the skeleton data
jointType = {'head': 0, 'shoulder_c': 1, 'spine': 2, 'hip_center': 3,
             'shoulder_l': 4, 'elbow_l': 5, 'wrist_l': 6, 'hand_l': 7,
             'shoulder_r': 8, 'elbow_r': 9, 'wrist_r': 10, 'hand_r': 11,
             'hip_l': 12, 'knee_l': 13, 'ankle_l': 14, 'foot_l': 15,
             'hip_r': 16, 'knee_r': 17, 'ankle_r': 18, 'foot_r': 19}


# New features
# For more information on why these new features you can refer to the document in the reference folder
distances_angles = ['hand_d', 'foot_d', 'hand_foot_l_d', 'hand_foot_r_d',
                    'head_hand_l_d', 'head_hand_r_d', 'elbow_l_a', 'elbow_r_a',
                    'knee_l_a', 'knee_r_a', 'hip_foot_r_d', 'hip_foot_l_d']

# acceleration and rotation data from inertial data
inertial = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']

# A list of all features name
columns = []
for i in jointType.keys():
    for j in ['x', 'y', 'z']:
        columns.append(i + '_' + j)

columns = columns + distances_angles + inertial

# call function to perform exponential weighted moving average and scaling to unit length on skeletons and inertial data
skeletons_dataset, inertial_dataset = ewma_scale_normalization(dataset, skeletons_dataset, inertial_dataset, jointType, columns, inertial)

# Create a empty Data frame with axis 0 = dataset and axis 1 = columns
all_df = pd.DataFrame(index=dataset, columns=columns)

# Fixed the length/size of each feature as maxlength
# Assign the normalized skeletons and inertial dataset to the empty data frame
maxlength = min([len(skeletons_dataset[s]['sk'][0][0]) for s in dataset])
for v in columns:
    if v in inertial:
        maxValue = max([max(abs(inertial_dataset[s][v])) for s in dataset])
        all_df[v] = [rescale(normalise(inertial_dataset[s][v], maxValue), maxlength) for s in dataset]
    else:
        maxValue = max([max(abs(skeletons_dataset[s][v])) for s in dataset])
        all_df[v] = [rescale(normalise(skeletons_dataset[s][v], maxValue), maxlength) for s in dataset]

# Add target
all_df['target'] = [i.split('_')[0][1:] for i in all_df.index]

# Save Processed Data
all_df.to_csv('all_df_new_features.csv')
pk.dump(all_df, open("all_df_new_features.pk", "wb"))
