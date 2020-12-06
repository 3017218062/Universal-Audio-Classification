from __init__ import *


class UacTransformer(object):
    def __init__(self):
        self.train_transform = Compose([
            Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
            Gain(min_gain_in_db=-5, max_gain_in_db=5, p=0.5),
            AddGaussianSNR(max_SNR=0.3, p=0.5),
        ])
        self.val_transform = None

    def __getitem__(self, item):
        if item == "train": return self.train_transform
        if item == "val": return self.val_transform
        return None
