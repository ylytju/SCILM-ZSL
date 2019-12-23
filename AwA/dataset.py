import numpy as np
from config import *


class Dataset(object):

    train_fea = np.array([])
    train_sem = np.array([])
    train_lab = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def _get_similar_pair(self):
        label = np.random.choice(self.unique_train_label)
        l, r = np.random.choice(self.map_train_label_indices[label],2,replace=True)
        return l, r, 1

    def _get_dissimilar_pair(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        l = np.random.choice(self.map_train_label_indices[label_l])
        r = np.random.choice(self.map_train_label_indices[label_r])
        return l, r, 0

    def _get_pair(self):
        if np.random.random() < 0.5:
            return self._get_similar_pair()
        else:
            return self._get_dissimilar_pair()

    def get_batch(self, n):  # n is the batch_size
        idex_left, idex_right, labels = [], [], []

        for _ in range(n):
            l,r,x = self._get_pair()
            idex_left.append(l)
            idex_right.append(r)
            labels.append(x)
        return self.att[idex_left,:], self.img[idex_right,:], np.expand_dims(labels, axis=1)


class LoadDataset(Dataset):
    def __init__(self, att, img, lab):
        self.img = img
        self.att = att
        self.lab = lab
        self.unique_train_label = np.unique(self.lab)
        self.map_train_label_indices = {label: np.flatnonzero(self.lab == label) for label in
                                        self.unique_train_label}

