import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import SGD, Adam

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.scores, self.scores_md, self.scores_total, self.out_score = [], [], [], []
        self.num_cls_per_task = args.num_cls_per_task # only makes sense if task id is known during training
        
        self.buffer = None

        self.statistics = {'mu': [], 'eigvec': [], 'eigval': []}

        self.args = args
        self.criterion = args.criterion # NLL
        self.net = args.net

        if self.args.optim_type == 'sgd':
            self.optimizer = SGD(self.net.parameters(), lr=self.args.lr)
        elif self.args.optim_type == 'adam':
            self.optimizer = Adam(self.net.parameters(), lr=self.args.lr)

        self.model_clip = args.model_clip

        # The actual label id from DataLoader.
        # This carries the label ids we've seen throughout training
        self.seen_ids = []
        # The actual label name from DataLoader.
        self.seen_names = []

        self.correct, self.til_correct, self.total, self.total_loss = 0., 0., 0., 0.
        self.cal_correct = 0.
        self.true_lab, self.pred_lab = [], []
        self.output_list, self.label_list = [], []

        self.saving_buffer = {}

    def observe(self, inputs, labels, not_aug_inputs=None, f_y=None, **kwargs):
        pass

    def evaluate(self, inputs, labels, task_id=None, **kwargs):
        self.net.eval()
        # labels = self.map_labels(labels)
        with torch.no_grad():
            out = self.net(inputs, self.args.normalize)
        pred = out.argmax(1)
        self.correct += pred.eq(labels).sum().item()
        self.total += len(labels)

        if task_id is not None:
            normalized_labels = labels % self.num_cls_per_task
            til_pred = out[:, task_id * self.num_cls_per_task:(task_id + 1) * self.num_cls_per_task]
            til_pred = til_pred.argmax(1)
            self.til_correct += til_pred.eq(normalized_labels).sum().item()

        self.net.train()

        if self.args.confusion:
            self.true_lab.append(labels.cpu().numpy())
            self.pred_lab.append(pred.cpu().numpy())

        if self.args.save_output:
            self.output_list.append(out.data.cpu().numpy())
            self.label_list.append(labels.data.cpu().numpy())

    def map_labels(self, labels):
        # labels: tensor
        relabel = []
        for y_ in labels:
            relabel.append(self.seen_ids.index(y_))
        return torch.tensor(relabel).to(self.args.device)

    def acc(self, reset=True):
        metrics = {}
        metrics['cil_acc'] = self.correct / self.total * 100
        metrics['til_acc'] = self.til_correct / self.total * 100
        if reset: self.reset_eval()
        return metrics

    def reset_eval(self):
        self.correct, self.til_correct, self.total, self.total_loss = 0., 0., 0., 0.
        self.true_lab, self.pred_lab = [], []
        self.output_list, self.label_list = [], []

    def save(self, **kwargs):
        """
            Save model specific elements required for resuming training
            kwargs: e.g. model state_dict, optimizer state_dict, epochs, etc.
        """
        raise NotImplementedError()

    def load(self, **kwargs):
        raise NotImplementedError()
