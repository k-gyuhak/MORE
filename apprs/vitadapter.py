import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from utils.sgd_hat import SGD_hat as SGD
from apprs.basemodel import BaseModel
from collections import Counter
from copy import deepcopy
from utils.utils import *
from torch.utils.data import DataLoader
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

class ViTAdapter(BaseModel):
    def __init__(self, args):
        super(ViTAdapter, self).__init__(args)
        self.scores, self.scores_md, self.scores_total, self.out_score = [], [], [], []
        self.feature_list, self.label_list = [], []
        self.p_mask, self.mask_back = None, None
        # self.net_list = []
        self.last_task_id = -1 # task id of lastly learned task
        self.cal_correct = 0.
        self.w, self.b = None, None
        self.cil_acc_mat_test = np.zeros((args.n_tasks + 1, args.n_tasks + 1)) - 100

        if args.distillation:
            self.criterion_distill = nn.KLDivLoss()

        if args.use_buffer:
            assert args.buffer_size
            if self.args.dataset in ['imagenet', 'timgnet']:
                self.buffer_dataset = Memory_ImageFolder(args)
            else:
                self.buffer_dataset = Memory(args)
        else:
            self.buffer_dataset = None

        self.ent = ComputeEnt(self.args)

    def observe(self, inputs, labels, names, not_aug_inputs=None, f_y=None, **kwargs):
        task_id = kwargs['task_id']
        b = kwargs['b']
        B = kwargs['B']
        s = self.update_s(b, B)

        n_samples = len(inputs)
        normalized_labels = labels % self.num_cls_per_task

        if self.buffer:
            try:
                inputs_bf, labels_bf = next(self.buffer_iter)
            except StopIteration:
                del self.buffer_iter
                self.buffer = DataLoader(self.buffer_dataset,
                                        batch_size=self.args.batch_size,
                                        sampler=self.sampler,
                                        num_workers=5,
                                        pin_memory=self.args.pin_memory)
                self.buffer_iter = iter(self.buffer)
                # self.buffer_iter = iter(self.buffer)
                inputs_bf, labels_bf = next(self.buffer_iter)

            inputs_bf = inputs_bf.to(device)
            # single dummy head
            labels_bf = torch.zeros_like(labels_bf).to(device) + self.num_cls_per_task
            normalized_labels_bf = labels_bf
            inputs = torch.cat([inputs, inputs_bf])
            labels = torch.cat([labels, labels_bf])
            normalized_labels = torch.cat([normalized_labels, normalized_labels_bf])

        features, masks = self.net.forward_features(task_id, inputs, s=s)
        outputs = self.net.forward_classifier(task_id, features)
        # outputs = outputs[:, task_id * self.num_cls_per_task:(task_id + 1) * self.num_cls_per_task]

        loss = self.criterion(outputs, normalized_labels)

        if self.args.distillation:
            self.teacher.eval()
            with torch.no_grad():
                outputs_t = self.teacher(inputs)
                outputs_t = outputs_t[:, task_id * self.num_cls_per_task:(task_id + 1) * self.num_cls_per_task]
            loss += self.criterion_distill(F.log_softmax(outputs / self.args.T, dim=1), 
                    F.softmax(outputs_t / self.args.T, dim=1)) * self.args.T * self.args.T * self.args.distill_lambda * self.num_cls_per_task

        loss += self.hat_reg(self.p_mask, masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.compensation(self.net, self.args.thres_cosh, s=s)

        hat = False
        if self.last_task_id >= 0:
            hat = True
        self.optimizer.step(hat=hat)
        self.compensation_clamp(self.net, self.args.thres_emb)

        self.total_loss = loss.item()
        outputs = outputs[:n_samples]
        scores, pred = outputs.max(1)
        self.scores.append(scores.detach().cpu().numpy())
        self.correct += pred.eq(normalized_labels[:n_samples]).sum().item()
        self.total += n_samples

        return loss.item()

    def train_clf(self, inputs, labels, model_ref, task_id, model_copy=None):
        T = 2
        self.net.train()

        # Must be SGD. SGD used for training the main network. When Adam is used for clf train, forgetting occured
        optim = SGD(self.net.head[task_id].parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # Check which samples are ID
        idx = labels // self.num_cls_per_task == task_id
        # Normalize
        labels[idx] = labels[idx] % self.num_cls_per_task
        # Assign dummy label
        labels[~idx] = self.num_cls_per_task

        outputs = self.net.forward_classifier(task_id, inputs)
        loss = self.criterion(outputs, labels)

        if model_copy is not None:
            prev_out = model_copy.forward_classifier(task_id, inputs)
            loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:, :-1] / self.args.T, dim=1), 
                            F.softmax(prev_out.detach()[:, :-1] / self.args.T)) * self.args.T * self.args.T * (prev_out.size(1) - 1)
            loss += self.args.distill_lambda * loss1

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.item()

    def train_embeddings(self, inputs, labels, task_id, **kwargs):
        b = kwargs['b']
        B = kwargs['B']
        s = self.update_s(b, B)

        self.net.train()

        # Check which samples are ID
        idx = labels // self.num_cls_per_task == task_id
        # Normalize
        labels[idx] = labels[idx] % self.num_cls_per_task
        # Assign dummy label
        labels[~idx] = self.num_cls_per_task

        outputs, _ = self.net(task_id, inputs, s=s)
        loss = self.criterion(outputs, labels)

        self.optimizer_ec.zero_grad()
        loss.backward()
        self.compensation(self.net, self.args.thres_cosh, s=s)
        self.modify_embeddings_grad(self.net, self.p_mask)
        self.optimizer_ec.step()

        return loss.item()

    def prepare_backward(self, p_task_id, task_id):
        params = [p for n, p in self.net.named_parameters() if f'ec1.{p_task_id}' in n or f'ec2.{p_task_id}' in n or f'head.{p_task_id}' in n]

        self.past_ebd, self.current_ebd = {}, {}
        for n, p in self.net.named_parameters():
            if f'ec1.{p_task_id}' in n or f'ec2.{p_task_id}' in n:
                self.past_ebd[n] = torch.sigmoid(self.args.smax * p.data)
            elif f'ec1.{task_id}' in n or f'ec2.{task_id}' in n:
                n_ = '.'.join(n.split('.')[:-1])
                self.current_ebd[n_] = torch.sigmoid(self.args.smax * p.data)

        if self.args.optim_type == 'sgd':
            self.optimizer_ec = SGD(params, lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optim_type == 'adam':
            raise NotImplementedError("HAT for Adam is not implemented")
            self.optimizer_ec = Adam(params, lr=self.args.lr)

    def prepare_block_backward(self, p_task_id, task_id, block):
        params = [p for n, p in self.net.named_parameters() if (f'ec1.{p_task_id}' in n or f'ec2.{p_task_id}' in n or f'head.{p_task_id}' in n) and f'blocks.{block}.' in n]
        for n, p in self.net.named_parameters():
            if (f'ec1.{p_task_id}' in n or f'ec2.{p_task_id}' in n or f'head.{p_task_id}' in n) and f'blocks.{block}.' in n:
                print(n)

        self.past_ebd, self.current_ebd = {}, {}
        for n, p in self.net.named_parameters():
            if f'ec1.{p_task_id}' in n or f'ec2.{p_task_id}' in n:
                self.past_ebd[n] = torch.sigmoid(self.args.smax * p.data)
            elif f'ec1.{task_id}' in n or f'ec2.{task_id}' in n:
                n_ = '.'.join(n.split('.')[:-1])
                self.current_ebd[n_] = torch.sigmoid(self.args.smax * p.data)

        if self.args.optim_type == 'sgd':
            self.optimizer_ec = SGD(params, lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optim_type == 'adam':
            raise NotImplementedError("HAT for Adam is not implemented")
            self.optimizer_ec = Adam(params, lr=self.args.lr)

    def modify_embeddings_grad(self, model, p_mask):
        for n, p in model.named_parameters():
            if n in self.past_ebd:
                n_ = '.'.join(n.split('.')[:-1])
                p.grad *= 1 - self.past_ebd[n]
                p.grad *= self.current_ebd[n_]
            # n_ = '.'.join(n.split('.')[:-1])
            # if n_ in p_mask:
            #     if p.grad is not None:
            #         p.grad *= p_mask[n_]

    def evaluate(self, inputs, labels, task_id, w=None, b=None, a=None, report_cil=True, **kwargs):
        """
        task_id is the task_id of provided data x, thus the correct id
        """
        if self.args.compute_auc or report_cil:
            total_learned_task_id = kwargs['total_learned_task_id']
        self.net.eval()
        self.total += len(labels)

        out_list, cal_output = [], []
        output_ood, cal_output_ood = [], []
        if report_cil:
            with torch.no_grad():
                entropy_list = []
                for t in range(total_learned_task_id + 1):
                    features, _ = self.net.forward_features(t, inputs, s=self.args.smax)
                    out = self.net.forward_classifier(t, features)
                    # out = out[:, t * self.num_cls_per_task:(t + 1) * self.num_cls_per_task]

                    if self.args.task_inference == 'entropy':
                        entropy_list.append(self.ent.compute(out))
                    else:
                        pass

                    out = F.softmax(out / 2, dim=1)
                    out = out[:, :self.num_cls_per_task]
                    # output_ood.append(out)
                    # Compute MD at prediction
                    if self.args.use_md:
                        md_list, dist = [], 0.
                        if len(self.args.cov_inv) > 0:
                            for y in range(t * self.num_cls_per_task, (t + 1) * self.num_cls_per_task):
                                cov_inv = self.args.cov_inv[t]
                                mean = self.args.mean[y]
                                # dist = md(features.detach().cpu().numpy(), mean, cov_inv, inverse=True) - dist # CHECK IF CORRECT
                                dist = md(features.detach().cpu().numpy(), mean, cov_inv, inverse=True) # CHECK IF CORRECT
                                if len(self.args.cov_inv_noise) > 0:
                                    cov_inv_noise = self.args.cov_inv_noise[t]
                                    dist = dist - 0.7 * md(features.detach().cpu().numpy(), mean, cov_inv_noise, inverse=True) # CHECK IF CORRECT
                                scores_md = 20 / dist
                                md_list.append(scores_md)
                            scores_md = np.concatenate(md_list, axis=1)
                            scores_md = scores_md.max(1, keepdims=True)
                            out = out * torch.from_numpy(scores_md).to(self.args.device)
                    if self.args.softmax:
                        out = F.softmax(out, dim=1)
                    out_list.append(out)
                    # output_ood.append(F.softmax(out, dim=1))
                    output_ood.append(out)

                    if w is not None:
                        cal_output.append(out * w[t] + b[t])
                        cal_output_ood.append(F.softmax(out * w[t] + b[t], dim=1))

                if len(entropy_list) > 0:
                    entropy_list = torch.cat(entropy_list, dim=-1)
                    task_id_pred = torch.min(entropy_list, dim=-1)[1]

            out_list = torch.cat(out_list, dim=1)
            output_ood = torch.cat(output_ood, dim=1)

            if len(entropy_list) > 0:
                # check if task_id_pred are correct
                true_tasks = labels // self.num_cls_per_task
                idx = task_id_pred == true_tasks
                # consider samples correctly predicted
                if sum(idx) == 0:
                    self.correct += 0
                else:
                    _, pred_cor = out_list[idx].max(1)
                    self.correct += pred_cor.eq(labels[idx]).sum().item()

                task_output_ood = []
                for task_pred, sample in zip(task_id_pred, output_ood):
                    task_output_ood.append(sample[task_pred * self.num_cls_per_task:(task_pred + 1) * self.num_cls_per_task].view(1, -1))
                output_ood = torch.cat(task_output_ood)
                total_scores, _ = output_ood.max(1)
                self.scores_total.append(total_scores.detach().cpu().numpy())
            else:
                _, pred = out_list.max(1)
                self.correct += pred.eq(labels).sum().item()

                total_scores, _ = output_ood.max(1)
                self.scores_total.append(total_scores.detach().cpu().numpy())

            if w is not None:
                cal_output = torch.cat(cal_output, dim=1)
                cal_output_ood = torch.cat(cal_output_ood, dim=1)

                _, cal_pred = cal_output.max(1)
                self.cal_correct += cal_pred.eq(labels).sum().item()
                scores, _ = cal_output_ood.max(1)
                # FIX ME: SCORES USING CALIBRATION OUTPUT IS NOT IMPLEMENTED
        else:
            self.correct += -1 * len(labels)
            self.cal_correct += -1 * len(labels)

        # Compute scores based on softmax using TIL heads
        # Compute scores based on MD using TIL heads though it doesn't matter if id not provided
        if task_id is not None:
            normalized_labels = labels % self.num_cls_per_task
            with torch.no_grad():
                features, _ = self.net.forward_features(task_id, inputs, s=self.args.smax)
                out = self.net.forward_classifier(task_id, features)
                out = F.softmax(out, dim=1)

                # Compute MD
                if self.args.compute_auc:
                    md_list, dist = [], 0
                    if len(self.args.cov.keys()) >= total_learned_task_id + 1:
                        md_for_clf = []
                        for y in range(task_id * self.num_cls_per_task, (task_id + 1) * self.num_cls_per_task):
                            cov_inv = self.args.cov_inv[task_id]
                            mean = self.args.mean[y]
                            # dist = md(features.detach().cpu().numpy(), mean, cov_inv, inverse=True) - dist
                            dist = md(features.detach().cpu().numpy(), mean, cov_inv, inverse=True)
                            if len(self.args.cov_inv_noise) > 0:
                                # mean_task = self.args.mean_task[task_id]
                                if len(self.args.cov_inv_noise) > 0:
                                    cov_inv_noise = self.args.cov_inv_noise[task_id]
                                    dist = dist - 0.7 * md(features.detach().cpu().numpy(), mean, cov_inv_noise, inverse=True) # CHECK IF CORRECT
                                # dist = dist - 0.8 * md(features.detach().cpu().numpy(), mean_task, cov_inv_noise)
                            scores_md = 20 / dist
                            md_for_clf.append(scores_md) # This is for Accuracy
                            md_list.append(-1 * dist) # This is for AUC MD
                        md_list = np.concatenate(md_list, axis=1)
                        scores = md_list.max(1) # dist_list is an array
                        self.scores_md.append(scores)
                        md_for_clf = np.concatenate(md_for_clf, axis=1)
                        md_for_clf = md_for_clf.max(1, keepdims=True)
                        out = out * torch.from_numpy(md_for_clf).to(self.args.device)

            # til_pred = out[:, task_id * self.num_cls_per_task:(task_id + 1) * self.num_cls_per_task]
            til_pred = out[:, :self.num_cls_per_task]
            scores, til_pred = til_pred.max(1)
            self.scores.append(scores.detach().cpu().numpy())
            self.til_correct += til_pred.eq(normalized_labels).sum().item()

            self.feature_list.append(features.data.cpu().numpy())
            self.label_list.append(labels.data.cpu().numpy())
        else:
            self.til_correct += -1 * len(labels)

        self.net.train()

        if self.args.confusion:
            self.true_lab.append(labels.cpu().numpy())
            self.pred_lab.append(pred.cpu().numpy())

        if self.args.save_output:
            self.output_list.append(out.data.cpu().numpy())

    def save(self, **kwargs):
        """
            Save model specific elements required for resuming training
            kwargs: e.g. model state_dict, optimizer state_dict, epochs, etc.
        """
        self.saving_buffer['buffer_dataset'] = self.buffer_dataset
        self.saving_buffer['w'] = self.w
        self.saving_buffer['b'] = self.b
        self.saving_buffer['p_mask'] = self.p_mask
        self.saving_buffer['mask_back'] = self.mask_back

        for key in kwargs:
            self.saving_buffer[key] = kwargs[key]

        torch.save(self.saving_buffer, self.args.logger.dir() + 'saving_buffer')

    def preprocess_task(self, **kwargs):
        # Add new embeddings for HAT
        self.net.append_embedddings()

        # Put label names in seen_names
        targets, names = zip(*sorted(zip(kwargs['loader'].dataset.targets,
                                         kwargs['loader'].dataset.names)))
        targets, names = list(targets), list(names)
        _, idx = np.unique(targets, return_index=True)
        for i in idx:
            self.seen_names.append(names[i])

        # Reset optimizer as there might be some leftover in optimizer
        if self.args.optim_type == 'sgd':
            self.optimizer = SGD(self.net.adapter_parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optim_type == 'adam':
            raise NotImplementedError("HAT for Adam is not implemented")
            self.optimizer = Adam(self.net.adapter_parameters(), lr=self.args.lr)

        # Prepare mask values for proper gradient update
        for n, p in self.net.named_parameters():
            p.grad = None
            if self.mask_back is not None:
                if n in self.mask_back.keys():
                    p.hat = self.mask_back[n]
                else:
                    p.hat = None
            else:
                p.hat = None

        # Prepare memory loader if memory data exist
        if self.args.use_buffer:
            if len(self.buffer_dataset.data) > 0:
                self.sampler = MySampler(len(self.buffer_dataset), len(kwargs['loader'].dataset))
                # We don't use minibatch. Use upsampling.
                self.buffer = DataLoader(self.buffer_dataset,
                                        batch_size=self.args.batch_size,
                                        sampler=self.sampler,
                                        num_workers=15,
                                        pin_memory=self.args.pin_memory)
                self.buffer_iter = iter(self.buffer)

    def end_task(self, task_id, **kwargs):
        self.last_task_id += 1
        assert self.last_task_id + 1 == task_id

        # Update masks for HAT
        self.p_mask = self.cum_mask(self.last_task_id, self.p_mask)
        self.mask_back = self.freeze_mask(self.last_task_id, self.p_mask)

        # Update memory if used
        if self.args.use_buffer and not self.args.train_clf:
            self.buffer_dataset.update(kwargs['train_loader'].dataset)

            self.args.logger.print(Counter(self.buffer_dataset.targets))

            if os.path.exists(self.args.logger.dir() + f'/memory_{self.last_task_id}'):
                self.args.logger.print("Memory exists. Not saving memory...")
            else:
                self.args.logger.print("Saving memory...")
                torch.save([deepcopy(self.buffer_dataset.data),
                            deepcopy(self.buffer_dataset.targets)],
                           self.args.logger.dir() + f'/memory_{self.last_task_id}')

    def acc(self, reset=True):
        metrics = {}
        metrics['cil_acc'] = self.correct / self.total * 100
        metrics['til_acc'] = self.til_correct / self.total * 100
        metrics['cal_cil_acc'] = self.cal_correct / self.total * 100
        if len(self.scores_total) > 0: metrics['scores_total'] = np.concatenate(self.scores_total)
        if len(self.scores) > 0: metrics['scores'] = np.concatenate(self.scores)
        if len(self.scores_md) > 0: metrics['scores_md'] = np.concatenate(self.scores_md)
        if reset: self.reset_eval()
        return metrics

    def reset_eval(self):
        self.correct, self.til_correct, self.total, self.total_loss = 0., 0., 0., 0.
        self.cal_correct = 0.
        self.true_lab, self.pred_lab = [], []
        self.output_list, self.label_list = [], []
        self.scores, self.scores_md, self.scores_total = [], [], []
        self.feature_list, self.label_list = [], []

    def update_s(self, b, B):
        """ b: current batch, B: total num batch """
        s = (self.args.smax - 1 / self.args.smax) * b / B + 1 / self.args.smax
        return s

    def compensation(self, model, thres_cosh=50, s=1):
        """ Equation before Eq. (4) in the paper """
        for n, p in model.named_parameters():
            if 'ec' in n:
                if p.grad is not None:
                    num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad *= self.args.smax / s * num / den

    def compensation_clamp(self, model, thres_emb=6):
        # Constrain embeddings
        for n, p in model.named_parameters():
            if 'ec' in n:
                if p.grad is not None:
                    p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

    def modify_grad(self, model, mask_back):
        """ 
            Zero-out gradients if both masks are 1. Eq. (2) in the paper
            Gradients of convolutions
        """
        for n, p in model.named_parameters():
            if n in mask_back:
                p.grad *= mask_back[n]

    def hat_reg(self, p_mask, masks):
        """ masks and self.p_mask must have values in the same order """
        reg, count = 0., 0.
        if p_mask is not None:
            for m, mp in zip(masks, p_mask.values()):
                aux = 1. - mp#.to(device)
                reg += (m * aux).sum()
                count += aux.sum()
            reg /= count
            return self.args.lamb1 * reg
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
            reg /= count
            return self.args.lamb0 * reg

    def cum_mask(self, t, p_mask):
        """ 
            Keep track of mask values. 
            This will be used later as a regularizer in the optimization
        """
        try:
            self.net = self.net.module
        except AttributeError:
            self.net = self.net

        task_id = torch.tensor([t]).to(device)
        mask = {}
        for n, _ in self.net.named_parameters():
            names = n.split('.')
            checker = [i for i in ['ec0', 'ec1', 'ec2'] if i in names]
            if names[0] == 'module':
                names = names[1:]
            if checker:
                if 'adapter' in n:
                    gc1, gc2 = self.net.__getattr__(names[0])[int(names[1])].__getattr__(names[2]).mask(task_id, s=self.args.smax)
                    if checker[0] == 'ec1':
                        n = '.'.join(n.split('.')[:-1])
                        mask[n] = gc1.detach()
                        mask[n].requires_grad = False
                    elif checker[0] == 'ec2':
                        n = '.'.join(n.split('.')[:-1])
                        mask[n] = gc2.detach()
                        mask[n].requires_grad = False

                elif checker[0] == 'ec0': # For ViT, there is no 'ec0', we can discard it
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = self.net.mask(task_id, self.args.smax).detach()
                    mask[n].requires_grad = False

        if p_mask is None:
            p_mask = {}
            for n in mask.keys():
                p_mask[n] = mask[n]
        else:
            for n in mask.keys():
                p_mask[n] = torch.max(p_mask[n], mask[n])
        return p_mask

    def freeze_mask(self, t, p_mask):
        """
            Eq (2) in the paper. self.mask_back is a dictionary whose keys are
            the convolutions' parameter names. Each value of a key is a matrix, whose elements are
            approximately binary.

            For ViT Adapter, there's only ec1 and ec2 in adapters in tranformer blocks.
            There are no other ec

            p_mask.keys() are [
            'blocks.0.adapter1.ec1', 'blocks.0.adapter1.ec2',
            'blocks.0.adapter2.ec1', 'blocks.0.adapter2.ec2',
            'blocks.1.adapter1.ec1', 'blocks.1.adapter1.ec2',
            'blocks.1.adapter2.ec1', 'blocks.1.adapter2.ec2',
            'blocks.2.adapter1.ec1', 'blocks.2.adapter1.ec2',
            'blocks.2.adapter2.ec1', 'blocks.2.adapter2.ec2',
            'blocks.3.adapter1.ec1', 'blocks.3.adapter1.ec2',
            'blocks.3.adapter2.ec1', 'blocks.3.adapter2.ec2',
            'blocks.4.adapter1.ec1', 'blocks.4.adapter1.ec2',
            'blocks.4.adapter2.ec1', 'blocks.4.adapter2.ec2',
            'blocks.5.adapter1.ec1', 'blocks.5.adapter1.ec2',
            'blocks.5.adapter2.ec1', 'blocks.5.adapter2.ec2',
            'blocks.6.adapter1.ec1', 'blocks.6.adapter1.ec2',
            'blocks.6.adapter2.ec1', 'blocks.6.adapter2.ec2',
            'blocks.7.adapter1.ec1', 'blocks.7.adapter1.ec2',
            'blocks.7.adapter2.ec1', 'blocks.7.adapter2.ec2',
            'blocks.8.adapter1.ec1', 'blocks.8.adapter1.ec2',
            'blocks.8.adapter2.ec1', 'blocks.8.adapter2.ec2',
            'blocks.9.adapter1.ec1', 'blocks.9.adapter1.ec2',
            'blocks.9.adapter2.ec1', 'blocks.9.adapter2.ec2',
            'blocks.10.adapter1.ec1', 'blocks.10.adapter1.ec2',
            'blocks.10.adapter2.ec1', 'blocks.10.adapter2.ec2',
            'blocks.11.adapter1.ec1', 'blocks.11.adapter1.ec2',
            'blocks.11.adapter2.ec1', 'blocks.11.adapter2.ec2'
            ]
        """
        try:
            self.net = self.net.module
        except AttributeError:
            self.net = self.net

        mask_back = {}
        for n, p in self.net.named_parameters():
            names = n.split('.')
            if 'adapter' in n: # adapter1 or adapter2. adapter.ec1, adapter.ec2
                # e.g. n is blocks.1.adapter1.fc1.weight
                if 'fc1.weight' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1, 1).expand_as(p)
                elif 'fc1.bias' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1)
                elif 'fc2.weight' in n:
                    post = p_mask['.'.join(names[:-2]) + '.ec2'].data.view(-1, 1).expand_as(p)
                    pre  = p_mask['.'.join(names[:-2]) + '.ec1'].data.view(1, -1).expand_as(p)
                    mask_back[n] = 1 - torch.min(post, pre)
                elif 'fc2.bias' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec2'].view(-1)
        return mask_back
