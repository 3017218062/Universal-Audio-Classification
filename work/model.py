from __init__ import *
from util import *

def cosineAnnealing(epoch, max_lr=3e-4, min_lr=5e-5, epochs=19):
    return ((math.cos(epoch * math.pi / (epochs - 1)) + 1.) * (max_lr - min_lr) / 2 + min_lr) / max_lr


class UacClassifier(pl.LightningModule):
    def __init__(self, classes=10, model="efficientnet_b0", pretrained=False, learning_rate=3e-4, weight_decay=1e-6, epochs=10, min_lr=1e-6, smooth=0., ohem=1.):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.min_lr = min_lr

        self.model = timm.create_model(model, pretrained=pretrained)
        if "dla60" in model:
            self.model.fc = Conv2d(self.model.fc.in_channels, classes, kernel_size=1, stride=1)
        elif "efficientnet" in model or "densenet" in model:
            self.model.classifier = Linear(in_features=self.model.classifier.in_features, out_features=classes)
        elif "regnet" in model or "rexnet" in model or "tresnet" in model or "csp" in model:
            self.model.head.fc = Linear(in_features=self.model.head.fc.in_features, out_features=classes)
        else:
            self.model.fc = Linear(in_features=self.model.fc.in_features, out_features=classes)

        self.train_criterion = CrossEntropyLoss(smooth=smooth, ohem=ohem)
        self.val_criterion = CrossEntropyLoss()

        self.train_metric = ClassificationMetric(recall=False, precision=False)
        self.val_metric = ClassificationMetric(recall=False, precision=False)

        self.history = {
            "loss": [], "acc": [], "f1": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        loss = self.train_criterion(_y, y)
        self.train_metric.update(_y, y)
        return loss

    def training_epoch_end(self, outs):
        loss = 0.
        for out in outs:
            loss += out["loss"].cpu().detach().item()
        loss /= len(outs)
        acc, f1 = self.train_metric.compute()

        self.history["loss"].append(loss)
        self.history["acc"].append(acc)
        self.history["f1"].append(f1)

    def validation_step(self, batch, idx):
        x, y = batch
        _y = self(x)
        val_loss = self.val_criterion(_y, y)
        self.val_metric.update(_y, y)
        return val_loss

    def validation_epoch_end(self, outs):
        val_loss = sum(outs).item() / len(outs)
        val_acc, val_f1 = self.val_metric.compute()

        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_f1"].append(val_f1)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: cosineAnnealing(epoch, max_lr=self.learning_rate, min_lr=self.min_lr, epochs=self.epochs), last_epoch=-1)
        return [optimizer], [scheduler]
