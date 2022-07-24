import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
import argparse
import IPython
import json
import scipy
from sklearn.metrics import f1_score
from collections import defaultdict
from deeprobust.graph.data import Dataset

from .graphsage_utils.encoders import Encoder
from .graphsage_utils.aggregators import MeanAggregator

class SupervisedGraphSageConcat(nn.Module):
    def __init__(self, num_classes, enc1, enc2, has_cuda):
        super(SupervisedGraphSageConcat, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.xent = nn.CrossEntropyLoss()
        self.cuda_ = has_cuda

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc1.embed_dim + enc2.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        self.enc2(nodes)
        embeds = torch.cat([self.enc1.forward_result, self.enc2.forward_result], axis=0)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        labels = labels.squeeze()
        if self.cuda_:
            labels=labels.cuda()
        return self.xent(scores, labels)

class SupervisedGraphSageConcat2(nn.Module):
    def __init__(self, num_classes, enc1, enc2, has_cuda):
        super(SupervisedGraphSageConcat2, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.xent = nn.CrossEntropyLoss()
        self.cuda_ = has_cuda

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc1.embed_dim + enc2.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        self.enc2(nodes)
        embeds = torch.cat([self.enc1.forward_result, self.enc2.forward_result], axis=0)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        labels = labels.squeeze()
        if self.cuda_:
            labels=labels.cuda()
        return self.xent(scores, labels)

class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc, has_cuda):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.cuda_ = has_cuda

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        labels = labels.squeeze()
        if self.cuda_:
            labels=labels.cuda()
        return self.xent(scores, labels)

def accuracy(output, labels):
    preds = output.detach().cpu().numpy().argmax(1)
    correct = (preds == labels.flatten())
    correct = correct.sum()
    return correct / len(labels)

class GraphSage():
    def __init__(self, num_classes, model_class="SupervisedGraphSage", hid_units=128,
                 gcn_encoder=True, gcn_aggregator=True, num_samples=[5, 5], cuda_=False, verbose=False):
        self.model_class = model_class
        self.num_classes = num_classes
        self.hid_units = hid_units
        self.gcn_encoder = gcn_encoder
        self.gcn_aggregator = gcn_aggregator
        self.num_samples = num_samples
        self.cuda = cuda_
        self.verbose = verbose
            
    def fit(self, feat_data, labels, adj_matrix, train, val,
            train_iters=200, patience=30, lr=0.7):
        self.lr = lr
        self.epochs = train_iters
        self.patience = patience
        
        np.random.seed(1)
        random.seed(1)
        num_nodes = feat_data.shape[0]
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(feat_data.toarray()), requires_grad=False)
        self.features = features
        self.num_nodes = num_nodes
        if self.cuda:
            features.cuda()
        
        adj_lists = dict()        
        cx = scipy.sparse.coo_matrix(adj_matrix)
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i not in adj_lists:
                adj_lists[i] = set([j])
            else:
                adj_lists[i].add(j)
        self.adj = adj_lists
        
        self.agg1 = MeanAggregator(features, cuda=self.cuda, gcn=self.gcn_aggregator)
        self.enc1 = Encoder(features, features.weight.shape[1], self.hid_units, adj_lists, self.agg1, gcn=self.gcn_encoder, cuda=self.cuda, num_sample=self.num_samples[0])
        self.agg2 = MeanAggregator(lambda nodes : self.enc1(nodes).t(), cuda=self.cuda, gcn=self.gcn_aggregator)
        self.enc2 = Encoder(lambda nodes : self.enc1(nodes).t(), self.enc1.embed_dim, self.hid_units, adj_lists, self.agg2,
            base_model=self.enc1, gcn=self.gcn_encoder, cuda=self.cuda, num_sample=self.num_samples[1])
        # self.enc1.num_sample = self.num_samples[0]
        # self.enc2.num_sample = self.num_samples[1]

        if self.model_class == "SupervisedGraphSageConcat":
            graphsage = SupervisedGraphSageConcat(self.num_classes, self.enc1, self.enc2, self.cuda)
        elif self.model_class == "SupervisedGraphSageConcat2":
            graphsage = SupervisedGraphSageConcat2(self.num_classes, self.enc1, self.enc2, self.cuda)
        else:
            graphsage = SupervisedGraphSage(self.num_classes, self.enc2, self.cuda)
        if self.cuda:
            graphsage.cuda()
        
        
        optimizer = torch.optim.SGD([p for p in graphsage.parameters() if p.requires_grad], lr=self.lr)
        times = []
        record_dict = dict()
        best_val_record_dict = None
        
        labels = labels.reshape(-1, 1)
        
        if self.verbose:
            print("Epoch Number\tLoss\tTrainAcc\tValAcc" )
        
        for batch in range(self.epochs):
            batch_nodes = train
            start_time = time.time()
            optimizer.zero_grad()
            batch_nodes_labels = Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
            if self.cuda:
                batch_nodes_labels.cuda()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)

            train_acc = accuracy(graphsage.forward(train), labels[train])
            val_acc = accuracy(graphsage.forward(val), labels[val])
            if self.verbose and batch % 20 == 0:
                print(batch, loss.data.cpu().numpy(), train_acc, val_acc)
            record_dict.update(dict(
                epoch=int(batch + 1), train_loss=float(loss.data), train_acc=float(train_acc),
                val_acc=float(val_acc), time=str(end_time-start_time), early_stopping=False
            ))

            if (best_val_record_dict is None) or (record_dict["val_acc"] >= best_val_record_dict["val_acc"]):
                best_val_record_dict = record_dict.copy()

        val_output = graphsage.forward(val)
        self.graphsage_model = graphsage
        if self.verbose:
            print("Validation F1:", f1_score(labels[val], val_output.data.cpu().numpy().argmax(axis=1), average="micro"))
            print("Average batch time:", np.mean(times))
            print(best_val_record_dict)
        
    def predict(self, adj=None, features=None):
        self.graphsage_model.eval()
        if adj is None and features is None:
            logits = self.graphsage_model.forward(np.arange(self.num_nodes))
            return torch.log_softmax(logits, dim=1)

        if adj is not None:
            adj_lists = dict()        
            cx = scipy.sparse.coo_matrix(adj)
            for i, j, v in zip(cx.row, cx.col, cx.data):
                if i not in adj_lists:
                    adj_lists[i] = set([j])
                else:
                    adj_lists[i].add(j)
            self.adj = adj_lists

        if features is not None:
            features_ = nn.Embedding(features.shape[0], features.shape[1])
            features_.weight = nn.Parameter(torch.FloatTensor(features.toarray()), requires_grad=False)
            self.features = features_
            if self.cuda:
                self.features.cuda()

        self.agg1.features = self.features
        self.enc1.features = self.features
        self.enc1.adj_lists = self.adj
        self.enc1.aggregator = self.agg1
        self.agg2.features = lambda nodes : self.enc1(nodes).t()
        self.enc2.features = lambda nodes : self.enc1(nodes).t()
        self.enc2.adj_lists = self.adj
        self.enc2.aggregator = self.agg2
        self.enc2.base_model = self.enc1

        if self.model_class == "SupervisedGraphSageConcat":
            self.graphsage_model.enc1 = self.enc1
            self.graphsage_model.enc2 = self.enc2
        elif self.model_class == "SupervisedGraphSageConcat2":
            self.graphsage_model.enc1 = self.enc1
            self.graphsage_model.enc2 = self.enc2
        else:
            self.graphsage_model.enc = self.enc2
        
        logits = self.graphsage_model.forward(np.arange(self.num_nodes))
        return torch.log_softmax(logits, dim=1) # This only covers evasion (post-training) setting.

    def eval(self):
        self.graphsage_model.eval()

if __name__ == '__main__':
	citeseer_dataset = Dataset(root='datasets/data', name='citeseer') 
	graphsage_class = GraphSage(num_classes = len(np.unique(citeseer_dataset.labels)), cuda_=True,
                                gcn_encoder=False, gcn_aggregator=False)
	graphsage_class.fit(citeseer_dataset.features, 
		citeseer_dataset.labels, 
		citeseer_dataset.adj, 
		citeseer_dataset.idx_train, 
		citeseer_dataset.idx_val, train_iters=25)
	graphsage_class.predict().detach()
	print("Passed unit test")
