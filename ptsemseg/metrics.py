# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import sklearn.metrics as metrics

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update_conf(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lt_flat = lt.flatten()
            lp_flat = lp.flatten()

            lp_flat = lp_flat[lt_flat!=250]
            lt_flat = lt_flat[lt_flat!=250]

            self.confusion_matrix += metrics.confusion_matrix(
                lt_flat, lp_flat, range(self.n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lt_flat = lt.flatten()
            lp_flat = lp.flatten()

            lp_flat = lp_flat[lt_flat!=250]
            lt_flat = lt_flat[lt_flat!=250]

            self.confusion_matrix += self._fast_hist(
                lt_flat, lp_flat, self.n_classes
            )
    def update_binary_oslsm(self, labels_true, preds):
        thresh = .5
        for lt, lp in zip(labels_true, preds):
            lt_flat = lt.flatten()
            lp_flat = lp.flatten()

            lp_flat = lp_flat[lt_flat!=250]
            lt_flat = lt_flat[lt_flat!=250]

            lp_flat = lp_flat>thresh
            tp = np.logical_and(lt_flat, lp_flat).sum()
            fp = np.logical_and(np.logical_not(lt_flat), lp_flat).sum()
            fn = np.logical_and(lt_flat, np.logical_not(lp_flat)).sum()
            iou = tp / (tp+fp+fn)
        return iou

    def update_binary(self, labels_true, label_preds):
        IOU = 0
        for lt, lp in zip(labels_true, label_preds):
            lt_flat = lt.flatten()
            lp_flat = lp.flatten()

            lp_flat = lp_flat[lt_flat!=250]
            lt_flat = lt_flat[lt_flat!=250]

            I = np.logical_and(lp_flat == 1, lt_flat == 1).sum()
            U = np.logical_or(lp_flat == 1, lt_flat == 1).sum()
            if U == 0:
                IOU = 1.0
            else:
                IOU += float(I) / U

        IOU = IOU / labels_true.shape[0]
        return IOU

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

