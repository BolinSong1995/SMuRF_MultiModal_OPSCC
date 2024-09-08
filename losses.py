import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class CoxLoss(_WeightedLoss):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    def forward(self, hazard_pred: torch.Tensor, survtime: torch.Tensor, censor: torch.Tensor,):

        current_batch_len = len(survtime)
        # modified for speed
        R_mat = survtime.reshape((1, current_batch_len)) >= survtime.reshape(
            (current_batch_len, 1)
        )
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
        )
        return loss_cox


class MMOLoss(nn.Module):
    def forward(self, embedding1, embedding2):
        return 0.5*(torch.max(1, torch.norm(embedding1, 'nuc')) + torch.max(1, torch.norm(embedding2, 'nuc'))) - torch.norm(torch.cat((embedding1, embedding2), dim=1), 'nuc')


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        gamma=0.5,
        criterion_class=nn.BCEWithLogitsLoss(),
        criterion_cox=CoxLoss()
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.criterion_class = criterion_class
        self.criterion_cox = criterion_cox

    def forward(self, task, pred_grade, pred_hazard, grade, time, event=None):
        if task == "multitask":
            grade_loss = self.criterion_class(pred_grade, grade)
            cox_loss = self.criterion_cox(pred_hazard, time, event)
            # print("losses: ", grade_loss, cox_loss)
            return self.gamma * grade_loss + (1 - self.gamma) * cox_loss
        elif task == "grade":
            grade_loss = self.criterion_class(pred_grade, grade)
            # print("losses: ", grade_loss)
            return grade_loss
        elif task == "survival":
            cox_loss = self.criterion_cox(pred_hazard, time, event)
            # print("losses: ", cox_loss)
            return cox_loss
        else:
            raise NotImplementedError(
                f'task method {task} is not implemented')
