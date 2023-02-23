import torch
import torch.nn as nn

import torch.nn.functional as F
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice_ce':
            return self.DiceCELoss
        elif mode == 'multilabel':
            return self.MultiClassLoss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        # loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def MultiClassLoss(self,logit, target):
        criterion=nn.BCEWithLogitsLoss()
        bceloss=criterion(logit,target)
        return bceloss




    def DiceCELoss(self, logit, target):
        logit=nn.Sigmoid()(logit)
        celoss=self.CrossEntropyLoss(logit,target)
        N = target.size(0)
        target = F.one_hot(target.to(torch.int64), 2)
        target = target.permute([0, 3, 1, 2])
        smooth = 1e-8
        input_flat = logit.contiguous().view(N,-1)
        target_flat = target.contiguous().view(N,-1)
        intersection = input_flat * target_flat
        diceloss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        diceloss= 1 - diceloss.sum() / N


        return celoss+diceloss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(2, 2, 7, 7).cuda()
    b = torch.rand(2,7,7).cuda()

    # print(loss.CrossEntropyLoss(a, b).item())
    print(loss.DiceCELoss(a, b))
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




