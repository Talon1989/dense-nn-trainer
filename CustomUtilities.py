import uuid

import numpy as np
import torch
import torch.nn as nn
from PySide6.QtCore import QRunnable


def one_hot(x):  # x has to be 1D numpy array, values has to be numerical
    cols = len(np.unique(x))
    Z = np.zeros([len(x), cols])
    for idx, val in enumerate(x):
        Z[idx, val] = 1
    return Z


class Model(nn.Module):
    def __init__(self, in_shape: int, hidden_shape: np.array, out_shape: int):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.module_list = nn.ModuleList()
        current_shape = in_shape
        for s in hidden_shape:
            self.module_list.append(nn.Linear(current_shape, s))
            self.module_list.append(nn.ReLU())
            current_shape = s
        self.output_layer = nn.Linear(current_shape, out_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        output = self.sigmoid(self.output_layer(x))
        return output


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4().hex

    def run(self):
        try:
            self.fn(self.args, self.kwargs)
        except Exception:
            self.fn()
        finally:
            print(f"{self.id} worker terminated")
