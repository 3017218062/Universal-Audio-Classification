from __init__ import *


class CSVLogger(Callback):
    def __init__(self, dirpath="history/", filename="history"):
        super(CSVLogger, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".csv":
            self.name += ".csv"

    def on_epoch_end(self, trainer, module):
        history = pd.DataFrame(module.history)
        history.to_csv(self.name, index=False)


class ModelCheckpoint(Callback):
    def __init__(self, dirpath="checkpoint/", filename="checkpoint", monitor="val_acc", mode="max"):
        super(ModelCheckpoint, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".pth":
            self.name += ".pth"
        self.monitor = monitor
        self.mode = mode
        self.value = 0. if mode == "max" else 1e6

    def on_epoch_end(self, trainer, module):
        if self.mode == "max" and module.history[self.monitor][-1] > self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(module.state_dict(), self.name)
        if self.mode == "min" and module.history[self.monitor][-1] < self.value:
            self.value = module.history[self.monitor][-1]
            torch.save(module.state_dict(), self.name)


class FlexibleTqdm(Callback):
    def __init__(self, steps, column_width=10):
        super(FlexibleTqdm, self).__init__()
        self.steps = steps
        self.column_width = column_width
        self.info = "\rEpoch_%d %s%% [%s]"

    def on_train_start(self, trainer, module):
        history = module.history
        self.row = "-" * (self.column_width + 1) * (len(history) + 2) + "-"
        title = "|"
        title += "epoch".center(self.column_width) + "|"
        title += "time".center(self.column_width) + "|"
        for i in history.keys():
            title += i.center(self.column_width) + "|"
        print(self.row)
        print(title)
        print(self.row)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
        current_index = int((batch_idx + 1) * 100 / self.steps)
        tqdm = ["."] * 100
        for i in range(current_index - 1):
            tqdm[i] = "="
        if current_index:
            tqdm[current_index - 1] = ">"
        print(self.info % (module.current_epoch, str(current_index).rjust(3), "".join(tqdm)), end="")

    def on_epoch_start(self, trainer, module):
        print(self.info % (module.current_epoch, "  0", "." * 100), end="")
        self.begin = time.perf_counter()

    def on_epoch_end(self, trainer, module):
        self.end = time.perf_counter()
        history = module.history
        detail = "\r|"
        detail += str(module.current_epoch).center(self.column_width) + "|"
        detail += ("%d" % (self.end - self.begin)).center(self.column_width) + "|"
        for j in history.values():
            detail += ("%.06f" % j[-1]).center(self.column_width) + "|"
        print("\r" + " " * 120, end="")
        print(detail)
        print(self.row)


class StochasticWeightAveraging(Callback):
    def __init__(self, dirpath="checkpoint/", filename="swa", start_epoch=0):
        super(StochasticWeightAveraging, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".pth":
            self.name += ".pth"
        self.start_epoch = start_epoch

    def on_epoch_end(self, trainer, module):
        n = module.current_epoch - self.start_epoch
        if n == 0:
            self.swa_weights = module.state_dict()
        elif n > 0:
            weights = module.state_dict()
            for i in self.swa_weights.keys():
                self.swa_weights[i] = self.swa_weights[i] * n + weights[i]
                if self.swa_weights[i].dtype == torch.int64:
                    self.swa_weights[i] //= (n + 1)
                elif self.swa_weights[i].dtype == torch.float32:
                    self.swa_weights[i] /= (n + 1)
                else:
                    pass
        else:
            pass

    def on_fit_end(self, trainer, module):
        torch.save(self.swa_weights, self.name)
        module.load_state_dict(self.swa_weights)


class LearningCurve(Callback):
    def __init__(self, dirpath="checkpoint/", filename="log", figsize=(12, 4), names=("loss", "acc", "f1")):
        super(LearningCurve, self).__init__()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.name = dirpath + filename
        if len(filename) > 4 and filename[-4:] != ".png":
            self.name += ".png"
        self.figsize = figsize
        self.names = names

    def on_fit_end(self, trainer, module):
        history = module.history
        plt.figure(figsize=self.figsize)
        for i, j in enumerate(self.names):
            plt.subplot(1, len(self.names), i + 1)
            plt.title(j + "/val_" + j)
            plt.plot(history[j], "--o", color='r', label=j)
            plt.plot(history["val_" + j], "-*", color='g', label="val_" + j)
            plt.legend()
        plt.savefig(self.name)
        plt.show()
