#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import random
import ipdb
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
import types
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import APPNP as APPNP_base
from deeprobust.graph import utils
from copy import deepcopy


class APPNP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, K=2, alpha=0.1, dropout=0.5, lr=0.01, weight_decay=5e-4,
             with_relu=True, with_bias=True, device=None):
        super(APPNP, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.K = K
        self.alpha = alpha
        self.with_relu = True
        self.with_bias = True
        self.output = None
        self.lr = lr
        self.weight_decay = weight_decay

        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNP_base(K, alpha)

        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=None):

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
        self.features = features.to_dense()
        self.adj = adj._indices()
        self.labels = labels

        data = types.SimpleNamespace()
        data.x = self.features
        data.edge_index = self.adj
        data.labels = self.labels

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(data)
            loss_train = F.nll_loss(output[idx_train], data.labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            loss_val = F.nll_loss(output[idx_val], data.labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], data.labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

    def predict(self, features=None, adj=None, idx=None):

         data = types.SimpleNamespace()
         if features is not None or adj is not None:
             features, adj = utils.to_tensor(features, adj, device=self.device)
             features = features.to_dense()
             adj = adj._indices()
             data.x = features
             data.edge_index = adj
         else:
             data.x = self.features
             data.edge_index = self.adj

         out = self.forward(data)
         if idx is not None:
             out = out[idx, :]
         self.output = out
         return out


if __name__ == '__main__':
     from deeprobust.graph.data import Dataset
     from modules.dataset import CustomDataset

     citeseer_dataset = CustomDataset(root='datasets/data', name='FB100')

     print(citeseer_dataset.features.shape[1])
     model = APPNP(nfeat=citeseer_dataset.features.shape[1], \
                   nclass=citeseer_dataset.labels.max().item()+1)

     model.fit(features=citeseer_dataset.features,
                      adj=citeseer_dataset.adj,
                      labels=citeseer_dataset.labels,
                      idx_train=citeseer_dataset.idx_train,
                      idx_val=citeseer_dataset.idx_val,
                      idx_test=citeseer_dataset.idx_test,
                      train_iters=200)

     model.eval()
     print(model.predict(idx=citeseer_dataset.idx_test))
