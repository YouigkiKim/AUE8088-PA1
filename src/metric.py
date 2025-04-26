from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('tn', default=torch.zeros(num_classes), dist_reduce_fx='sum')
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1).view(-1)
        target = target.view(-1)
        for cls in range(self.num_classes):
            pred_is_cls = preds == cls
            true_is_cls = target == cls
            tp = torch.sum(pred_is_cls & true_is_cls)
            fp = torch.sum(pred_is_cls & ~true_is_cls)
            fn = torch.sum(~pred_is_cls & true_is_cls)
            tn = torch.sum(~pred_is_cls & ~true_is_cls)
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn
            self.tn[cls] += tn
    def compute(self):
        precision = self.tp.float() / (self.tp + self.fp).clamp(min=1)
        recall = self.tp.float() / (self.tp + self.fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        return f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds   = torch.argmax(preds, dim=1)
        target  = target.view(-1)
        preds   = preds.view_as(target)
        # [TODO] check if preds and target have equal shape
        correct = torch.sum(preds==target)
        # [TODO] Cound the number of correct prediction
        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
