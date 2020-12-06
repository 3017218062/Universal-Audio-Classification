from __init__ import *


class CrossEntropyLoss(Module):
    def __init__(self, smooth=0., ohem=1.):
        super().__init__()
        self.smooth = smooth
        self.ohem = ohem

    def forward(self, input, target, reduction="mean"):
        N, C = input.size()

        if target.dim() > 1:
            one_hot = target
        else:
            one_hot = torch.zeros((N, C), dtype=input.dtype, device=input.device)
            one_hot.scatter_(1, target.reshape(N, 1), 1)
        if self.smooth > 0:
            one_hot = (1. - self.smooth) * one_hot + (self.smooth / C)

        loss = -(one_hot * F.log_softmax(input, 1)).sum(1)
        if self.ohem < 1:
            loss, _ = loss.topk(k=int(N * self.ohem))

        if reduction == "mean":
            return loss.mean(0)
        elif reduction == "sum":
            return loss.sum(0)
        else:
            return loss


class ClassificationMetric(object):
    def __init__(self, accuracy=True, recall=True, precision=True, f1=True, average="macro"):
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        self.f1 = f1
        self.average = average

        self.preds = []
        self.target = []

    def reset(self):
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target):
        preds = list(preds.cpu().detach().argmax(1).numpy())
        target = list(target.cpu().detach().argmax(1).numpy()) if target.dim() > 1 else list(target.cpu().detach().numpy())
        self.preds += preds
        self.target += target

    def compute(self):
        metrics = []
        if self.accuracy:
            metrics.append(accuracy_score(self.target, self.preds))
        if self.recall:
            metrics.append(recall_score(self.target, self.preds, labels=list(set(self.preds)), average=self.average))
        if self.precision:
            metrics.append(precision_score(self.target, self.preds, labels=list(set(self.preds)), average=self.average))
        if self.f1:
            metrics.append(f1_score(self.target, self.preds, labels=list(set(self.preds)), average=self.average))
        self.reset()
        return metrics


def metrics_print(yTrue, yPred, classes):
    print(classification_report(yTrue, yPred, target_names=classes, digits=4))


def cm_draw(yTrue, yPred, classes):
    cm = confusion_matrix(yTrue, yPred, labels=range(len(classes)))
    df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(len(classes) + 3, len(classes) + 3))
    sns.heatmap(df, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.show()
