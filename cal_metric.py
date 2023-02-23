import numpy as np
import  cv2
import os
def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    # print(iu)
    # mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    print("precision:{},recall:{},F1:{},iou:{},OA:{}".format(precision[1],recall[1],F1[1],iu[1],acc))
    # score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}

    # return score_dict

def get_confuse_matrix(num_classes, label_gts_path, label_preds_path,names):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for name in names:
        lt=cv2.imread(((os.path.join(label_gts_path,name))),0)
        lt = lt.clip(max=1)
        lp=cv2.imread(((os.path.join(label_preds_path,name))),0)
        lp = lp.clip(max=1)
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix

if __name__ == '__main__':
    label_gts_path=r'...\LEVIR-CD\test\label'
    label_preds_path=r'...\LEVIR'
    names=os.listdir(label_preds_path)
    print(len(names))
    num_classes=2
    confusion_matrix=get_confuse_matrix(num_classes, label_gts_path, label_preds_path,names)
    cm2score(confusion_matrix)

