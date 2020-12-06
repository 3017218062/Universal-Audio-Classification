from __init__ import *
from dataset import *

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("-b", "--batch_size", default=256, type=int)
parser.add_argument("-t", "--target_size", default=224, type=int)
parser.add_argument("-m", "--model", default="dla60_res2next", type=str)
parser.add_argument("-f", "--folds", default=5, type=int)
parser.add_argument("-a", "--tta", default='y', type=str)
args = parser.parse_args()


class TestDataset(Dataset):
    def __init__(self, filenames, n_mels=64, transform=None):
        self.filenames = filenames
        self.n_mels = n_mels
        self.transform = transform
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sound, sr = librosa.load(input_path + "test/" + self.filenames[idx], sr=16000)
        sound = CropOrPad(sound, sr)

        if self.transform:
            sound = self.transform(sound, sr)

        melspec = Wav2Melspectrogram(sound, sr, self.n_mels)
        melspec = Normalize(melspec)
        image = np.stack([melspec, melspec, melspec], axis=-1)
        image = cv2.resize(image, (self.n_mels, self.n_mels), interpolation=cv2.INTER_CUBIC)

        image = torch.from_numpy(image.transpose((2, 0, 1))).type(torch.float)
        return image


class TestModel(Module):
    def __init__(self, classes, model):
        super().__init__()
        self.model = timm.create_model(model, pretrained=False)
        if "dla60" in model:
            self.model.fc = Conv2d(self.model.fc.in_channels, classes, kernel_size=1, stride=1)
        elif "efficientnet" in model or "densenet" in model:
            self.model.classifier = Linear(in_features=self.model.classifier.in_features, out_features=classes)
        elif "regnet" in model or "rexnet" in model or "tresnet" in model or "csp" in model:
            self.model.head.fc = Linear(in_features=self.model.head.fc.in_features, out_features=classes)
        else:
            self.model.fc = Linear(in_features=self.model.fc.in_features, out_features=classes)

    def forward(self, x):
        return self.model(x)


input_path = "../../data-tmp/"
output_path = "../output/"
device = "cuda" if torch.cuda.is_available() else "cpu"
submission = pd.read_csv(input_path + "submission.csv")
class_map = os.listdir(input_path + "train/")
class_map = dict(zip(range(len(class_map)), class_map))
classes = len(class_map)
sub = None

test_dataset = TestDataset(submission["file_name"], n_mels=(args.target_size, args.target_size)[0])
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
for i in range(args.folds):
    print("Fold-%d start..." % i)
    model = TestModel(classes, args.model)
    model.load_state_dict(torch.load(output_path + "checkpoint_%s_fold%d.pth" % (args.model, i), map_location="cpu"))
    model = model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for x in tqdm(test_dataloader):
            y = model(x.to(device)).cpu().numpy()
            outputs.append(y)
    if sub is None:
        sub = np.vstack(outputs)
    else:
        sub += np.vstack(outputs)
    del model
del test_dataset, test_dataloader
gc.collect()

if args.tta == 'y':
    transforms = [
        Shift(min_fraction=-0.1, max_fraction=-0.099999, p=1.),
        Shift(min_fraction=0.099999, max_fraction=0.1, p=1.),
        Gain(min_gain_in_db=4.999999, max_gain_in_db=5, p=1.),
        Gain(min_gain_in_db=-5, max_gain_in_db=-4.999999, p=1.),
    ]
    for t in transforms:
        test_dataset = TestDataset(submission["file_name"], n_mels=(args.target_size, args.target_size)[0], transform=t)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        for i in range(5):
            print("Fold-%d start..." % i)
            model = TestModel(classes, args.model)
            model.load_state_dict(torch.load(output_path + "checkpoint_%s_fold%d.pth" % (args.model, i), map_location="cpu"))
            model = model.to(device)
            model.eval()
            outputs = []
            with torch.no_grad():
                for x in tqdm(test_dataloader):
                    y = model(x.to(device)).cpu().numpy()
                    outputs.append(y)
            sub += np.vstack(outputs)
            del model
        del test_dataset, test_dataloader
        gc.collect()

submission["label"] = np.argmax(sub, axis=1)
submission["label"] = submission["label"].apply(lambda x: class_map[x])
submission.to_csv(output_path + "submission.csv", index=False)
print(submission)
