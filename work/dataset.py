from __init__ import *


def Wav2Melspectrogram(sound, sr, n_mels):
    melspec = librosa.feature.melspectrogram(sound, sr=sr, n_mels=n_mels)
    melspec = librosa.power_to_db(melspec)
    return melspec


def CropOrPad(sound, l):
    padding = l - len(sound)
    if padding > 0:
        sound = np.hstack([sound, np.zeros((padding,))])
    return sound


def Normalize(image):
    MAX, MIN = np.max(image), np.min(image)
    if MAX - MIN > 1e-6:
        image = (image - MIN) / (MAX - MIN)
    return image


class UacDataset(Dataset):
    def __init__(self, filenames, labels, n_mels=64, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.n_mels = n_mels
        self.transform = transform
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sound, sr = librosa.load(self.filenames[idx], sr=16000)
        sound = CropOrPad(sound, sr)

        if self.transform:
            sound = self.transform(sound, sr)

        melspec = Wav2Melspectrogram(sound, sr, self.n_mels)
        melspec = Normalize(melspec)
        image = np.stack([melspec, melspec, melspec], axis=-1)
        image = cv2.resize(image, (self.n_mels, self.n_mels), interpolation=cv2.INTER_CUBIC)

        image = torch.from_numpy(image.transpose((2, 0, 1))).type(torch.half)

        label = self.labels[idx]
        return image, label
