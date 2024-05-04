"""
=================================================
coding:utf-8
@Time:      2024/5/4 21:00
@File:      within_CR.py
@Author:    Ziwei Wang
@Function:
=================================================
"""
import mne
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import random
from utils.alg_utils import EA, EA_online
from utils.aug_utils import leftrightflipping_transform
from scipy.linalg import fractional_matrix_power

def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    elif dataset == 'MI1-7':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)

        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        trials_arr = np.array([[120, 120, 160, 160, 160],
                    [120, 120, 160, 120, 160],
                    [120, 120, 160, 160, 160],
                    [120, 140, 160, 160, 160],
                    [120, 140, 160, 160, 160],
                    [120, 120, 160, 160, 160],
                    [120, 120, 160, 160, 160],
                    [160, 120, 160, 160, 160],
                    [120, 120, 160, 160, 160]])

        # only use session 1's first 120 trials, remove session 4-5
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(120) + np.sum(trials_arr[:i, :]))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif 'MI1' in dataset:
        paradigm = 'MI'
        num_subjects = 5
        if dataset == 'MI1-7':
            num_subjects = 7
        sample_rate = 1000
        ch_num = 59
    elif 'BNCI2014008' in dataset:
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


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


def data_alignment_returnref(x):
    '''
    :param X: np array, EEG data
    :return: np array, aligned EEG data; np array, reference matrix
    '''
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA, refEA


def data_alignment_passref(x, pre_refEA, num):
    '''
    :param X: np array, EEG data; np array, calcualted reference matrix; int, number of pre-data
    :return: np array, aligned EEG data
    '''
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    refEA = (refEA * len(x) + pre_refEA * num) / (len(x) + num)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def traintest_split_within_subject(dataset, X, y, num_subjects, test_subject_id, num, shuffle):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    class_out = len(np.unique(subj_label))
    if shuffle:
        inds = np.arange(len(subj_data))
        np.random.shuffle(inds)
        subj_data = subj_data[inds]
        subj_label = subj_label[inds]
    if num < 1:  # percentage
        num_int = int(len(subj_data) * num / class_out)
    else:  # numbers
        num_int = int(num)

    inds_all_train = []
    inds_all_test = []
    for class_num in range(class_out):
        inds_class = np.where(subj_label == class_num)[0]
        inds_all_train.append(inds_class[:num_int])
        inds_all_test.append(inds_class[num_int:])
    inds_all_train = np.concatenate(inds_all_train)
    inds_all_test = np.concatenate(inds_all_test)

    train_x = subj_data[inds_all_train]
    train_y = subj_label[inds_all_train]
    test_x = subj_data[inds_all_test]
    test_y = subj_label[inds_all_test]

    print('Within subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def ml_within(dataset, align, approach, calbr_nperclass):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if 'MI1' in dataset:
        print('MI downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)

    print('sample rate:', sample_rate)
    '''
    print(X[0, :, 0])
    if dataset == 'BNCI2014001':
        adj_mat = [[0, 2], [0, 3], [0, 4], [1, 2], [1, 6], [1, 7], [1, 8], [2, 3], [2, 7], [2, 8], [2, 9], [3, 4],
                   [3, 8], [3, 9], [3, 10], [4, 5], [4, 9], [4, 10], [4, 11], [5, 10], [5, 11], [5, 12], [6, 7],
                   [6, 13], [7, 8], [7, 13], [7, 14], [8, 9], [8, 13], [8, 14], [8, 15], [9, 10], [9, 14], [9, 15],
                   [9, 16], [10, 11], [10, 15], [10, 16], [10, 17], [11, 12], [11, 16], [11, 17], [12, 17], [13, 14],
                   [13, 18], [14, 15], [14, 18], [14, 19], [15, 16], [15, 18], [15, 19], [15, 20], [16, 17], [16, 19],
                   [16, 20], [17, 20], [18, 19], [18, 21], [19, 20], [19, 21], [20, 21]]
    X = small_laplace_normalize(X, adj_mat)
    '''

    scores_arr = []

    for i in range(num_subjects):
        # if paradigm == 'MI':
        train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, calbr_nperclass, False)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        # elif paradigm == 'ERP':
        #     train_x, test_x = X[:calbr_nperclass], X[calbr_nperclass:]
        #     train_y, test_y = y[:calbr_nperclass], y[calbr_nperclass:]
        #     print('ERP: train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape,
        #           test_y.shape)
        if align:
            # continous EA
            train_x, R = data_alignment_returnref(train_x)

            num_samples = len(train_x)
            test_x_aligned = []
            for ind in range(len(test_y)):
                curr = test_x[ind]
                R = EA_online(curr, R, num_samples)
                num_samples += 1
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                curr_aligned = np.dot(sqrtRefEA, curr)
                test_x_aligned.append(curr_aligned)
            test_x = np.stack(test_x_aligned)
        if paradigm == 'MI':
            # CSP
            csp = mne.decoding.CSP(n_components=10)

            # mirroring
            if dataset == 'BNCI2014001':
                left_mat = [1, 2, 6, 7, 8, 13, 14, 18]
                right_mat = [5, 4, 12, 11, 10, 17, 16, 20]
                aug_train_x = leftrightflipping_transform(
                    torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                    left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                aug_train_y = 1 - train_y
            elif dataset == 'BNCI2014002':
                left_mat = [0, 3, 4, 5, 6, 12]
                right_mat = [2, 11, 10, 9, 8, 14]
                aug_train_x = leftrightflipping_transform(
                    torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                    left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                aug_train_y = train_y
            elif dataset == 'BNCI2014004':
                left_mat = [0]
                right_mat = [2]
                aug_train_x = leftrightflipping_transform(
                    torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                    left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                aug_train_y = 1 - train_y
            elif dataset == 'BNCI2015001':
                left_mat = [0, 3, 4, 5, 10]
                right_mat = [2, 9, 8, 7, 12]
                aug_train_x = leftrightflipping_transform(
                    torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                    left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                aug_train_y = train_y
            elif dataset == 'PhysionetMI':
                left_mat = [0, 1, 2, 21, 24, 25, 29, 30, 31, 32,
                            14, 15, 16, 38, 40, 42, 44, 46,
                            7, 8, 47, 48, 49,
                            55, 56, 60]
                right_mat = [6, 5, 4, 23, 28, 27, 37, 36, 35, 34,
                             20, 19, 18, 39, 41, 43, 45, 54,
                             13, 12, 53, 52, 51,
                             59, 58, 62]
                aug_train_x = leftrightflipping_transform(
                    torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                    left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                aug_train_y = train_y
            elif 'MI1' in dataset:
                if i == 0 or i == 5:
                    # for non-leftright subjects, no reflection
                    aug_train_x, aug_train_y = train_x, train_y
                else:
                    left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49, 50, 55, 57]
                    right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53, 52, 56, 58]
                    aug_train_x = leftrightflipping_transform(
                        torch.from_numpy(train_x).to(torch.float32).reshape(train_x.shape[0], 1, ch_num, -1),
                        left_mat, right_mat).numpy().reshape(train_x.shape[0], ch_num, -1)
                    aug_train_y = 1 - train_y

            train_x = np.concatenate((train_x, aug_train_x))
            train_y = np.concatenate((train_y, aug_train_y))

            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)

            # classifier
            pred, model = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
            score = np.round(accuracy_score(test_y, pred), 5)
            print('score', np.round(score, 5))
        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100, 3)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))
    return scores_arr

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    dataset_arr = ['BNCI2014001']
    # how many training labeled samples for one class
    clbr_num_arr = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # 5, 10, 15, 20, 25, 30, 35, 40, 45
    scores = np.zeros((len(dataset_arr), len(clbr_num_arr)))
    cnt0 = 0
    for dataset in dataset_arr:
        cnt1 = 0
        for approach in ['LDA']:
            # use EA or not
            align = True
            print(dataset, align, approach)
            for calbr_nperclass in clbr_num_arr:
                scores_arr = ml_within(dataset, align, approach, calbr_nperclass)
                score_avg = np.round(np.average(scores_arr), 2)
                print(score_avg)
                scores[cnt0, cnt1] = score_avg
                cnt1 += 1
        cnt0 += 1

    print(scores)