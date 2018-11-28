import scipy.io as sio


# Create a function to load inertial and skeleton data
# For more information on the data can go to http://www.utdallas.edu/~cxc123730/UTD-MHAD.html


def data_load():
    # Create a list of filename of the data files available
    action_id = ['a'+str(i) for i in range(1, 28)]  # total 27 actions
    subject_id = ['s'+str(i) for i in range(1, 9)]  # 8 subjects
    rep_id = ['t'+str(i) for i in range(1, 5)]  # 4 repetition
    file_namelist = ['_'.join([a, s, r]) for s in subject_id for a in action_id for r in rep_id]

    # Remove the corrupted data file names
    file_namelist.remove('a8_s1_t4')
    file_namelist.remove('a23_s6_t4')
    file_namelist.remove('a27_s8_t4')

    # the inertial data is in .mat format therefore it required scipy.io library to load the data
    # sampling rate is 50Hz per action
    # Time frame per action varies depending on the start frame and end frame of the action
    # d_iner is the name of the 2D data

    inertial_data = {
        file_name: {
            'inertial': sio.loadmat('./Inertial/' + file_name + '_inertial.mat')['d_iner']}
        for file_name in file_namelist}

    # the skeletons data is in .mat format therefore it required scipy.io library to load the data
    # each input skeleton sequence is denoted as a T x N x 3 matrix
    # T is the total number of frames
    # 30 frames per second
    # Time frame per action varies depending on the start frame and end frame of the action
    # N is the number of joints
    # 3 indicates x, y and z coordinates for each joint
    # d_skel is the name of the 3D data

    skeletons_data = {
        file_name: {
            'sk': sio.loadmat('./Skeleton/' + file_name + '_skeleton.mat')['d_skel']}
        for file_name in file_namelist}

    return file_namelist, inertial_data, skeletons_data
