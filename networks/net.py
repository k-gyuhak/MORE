import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import clip

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias

        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

        self.seen_classes = []

    def forward(self, x, normalize=False):
        # x must be normalized
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
            unit_w = self.fc.weight / self.fc.weight.norm(dim=-1, keepdim=True)
            out = 100 * x @ unit_w.T
        else:
            out = self.fc(x)
        return out

    def make_head(self, new_dim, c, clip_init=None):
        # new_dim: size of dimension to add, must be 1. c: class name
        # clip_init: a pretrained clip model
        device = self.fc.weight.device
        if c not in self.seen_classes:
            self.seen_classes.append(c)

            # self.total_dim = self.out_dim + new_dim
            self.total_dim = len(self.seen_classes)
            self.fc1 = deepcopy(self.fc)

            self.fc = nn.Linear(self.in_dim, self.total_dim, self.bias).to(device)
            self.fc.weight.data[:self.out_dim, :] = self.fc1.weight.data

            if clip_init is not None:
                print("a photo of a {}".format(c))
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}")]).to(device)
                text_feature = clip_init.encode_text(text_inputs).type(torch.FloatTensor).to(device)
                self.fc.weight.data[-1, :] = text_feature.data

            if self.bias:
                self.fc.bias.data[:self.out_dim] = self.fc1.bias.data

            # self.out_dim = self.out_dim + new_dim
            self.out_dim = self.total_dim
            del self.fc1

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)
