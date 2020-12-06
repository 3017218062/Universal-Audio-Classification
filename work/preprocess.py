from __init__ import *


class TrainValSplit(object):
    def __init__(self, path, folds=5, fold=0, one_hot=False, seed=2020):
        self.classes = os.listdir(path)
        self.classes = dict(zip(self.classes, range(len(self.classes))))

        filenames, labels = [], []
        for i, j in self.classes.items():
            tmp = os.listdir(path + i)
            for k in tmp:
                filenames.append(path + i + "/" + k)
                labels.append(j)
        filenames, labels = np.asarray(filenames), np.asarray(labels)

        split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(filenames, labels)
        if one_hot:
            labels = np.eye(len(self.classes))[labels]
        for i in range(fold): next(split)
        train_indices, val_indices = next(split)

        self.train_len, self.val_len = len(train_indices), len(val_indices)
        self.train_filenames, self.train_labels = filenames[train_indices], labels[train_indices]
        self.val_filenames, self.val_labels = filenames[val_indices], labels[val_indices]
        del filenames, labels, split, train_indices, val_indices
        gc.collect()

    def __getitem__(self, idx):
        if idx == "train": return self.train_filenames, self.train_labels
        if idx == "val": return self.val_filenames, self.val_labels
        if idx == "class": return self.classes
        return None
