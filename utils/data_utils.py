import numpy as np
from utils.alg_utils import EA


def traintest_split_cross_subject(X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    X_t = data_subjects.pop(test_subject_id)
    y_t = labels_subjects.pop(test_subject_id)
    X_s = np.concatenate(data_subjects)
    y_s = np.concatenate(labels_subjects)
    return X_s, y_s, X_t, y_t


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X
