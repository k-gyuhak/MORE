# This code uses self.label = [] and save the super-label and sub-label of a sample.

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'full_labels', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, full_labels: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                if attr_str == 'logits':
                    setattr(self, attr_str, [])
                else:
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, full_labels=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, full_labels, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if full_labels is not None:
                    self.full_labels[index] = full_labels[i].to(self.device)
                    # if self.num_seen_examples > self.buffer_size:
                    #     self.full_labels[index] == full_labels[i].to(self.device)
                    # else:
                        # self.full_labels.append(full_labels[i].to(self.device))
                if logits is not None:
                    # self.logits[index] = logits[i].to(self.device)
                    if self.num_seen_examples > self.buffer_size:
                        self.logits[index] = logits[i].to(self.device)
                    else:
                        self.logits.append(logits[i].to(self.device))
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, mse=None, ce=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        n_size = min(self.num_seen_examples, self.examples.size(0))
        if mse is not None:
            mse_copy = mse.clone()
            for i in range(len(mse_copy)):
                drop = torch.rand(len(mse_copy[i])) < 0.1
                mse_copy[i][drop] = 0
            dists = torch.cdist(mse_copy, self.examples[:n_size])
            choice = dists.argmax(1)
            print('mse', choice, len(set(choice.cpu().numpy())))
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                      size=size, replace=False)
        elif ce is not None:
            ce_copy = ce.clone()
            for i in range(len(ce_copy)):
                drop = torch.rand(len(ce_copy[i])) < 0.1
                ce_copy[i][drop] = 0
            dists = torch.cdist(ce_copy, self.examples[:n_size])
            choice = dists.argmin(1)
            print('ce', choice, len(set(choice.cpu().numpy())))
        else:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                      size=size, replace=False)

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if attr_str == 'logits':
                    ret_tuple += ([attr[c] for c in choice],)
                else:
                    ret_tuple += (attr[choice],)
        ret_tuple += (choice, )
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def reduce(self, new_size):
        # print(new_size, len(self.examples))
        # if new_size >= self.buffer_size or new_size > len(self.logits):
        #     pass
        # # else:
        if new_size < self.buffer_size and new_size < len(self.logits):
            perm = np.random.permutation(self.buffer_size)[:new_size]
            self.buffer_size = new_size

            self.examples = self.examples[perm]
            self.logits = [self.logits[p] for p in perm]
            self.labels = self.labels[perm]
            self.full_labels = self.full_labels[perm]

            # for attr_str in self.attributes:
            #     if hasattr(self, attr_str):
            #         attr = getattr(self, attr_str)
            #         if attr_str == 'logits':
            #             attr = [attr[p] for p in perm]
            #         else:
            #             attr = attr[perm]



