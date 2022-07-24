import torch as th
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch.nn.functional import normalize
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv

class JKNet(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers=6, n_units=32, dropout=0.5,
                 activation="relu", operation='Maxpool', use_cuda=False):
        super(JKNet, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.use_cuda = use_cuda
        assert operation in ['Maxpool', 'ConCat']
        self.operation = operation
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(in_feats, n_units, activation=self.activation))
        self.dropout = dropout
        for i in range(1, self.n_layers):
            self.layers.append(
                GraphConv(n_units, n_units, activation=self.activation))
        if self.operation == 'Maxpool':
            self.layers.append(GraphConv(n_units, out_feats))
        elif self.operation == 'ConCat':
            self.layers.append(GraphConv(n_layers * n_units, out_feats))
        else:
            raise Exception("Please choose operation from [Maxpool, ConCat]")

    def evaluate(self, features, g, labels, mask):
        self.eval()
        with th.no_grad():
            logits = self.forward(features, g)
            logits = logits[mask]
            _, indices = th.max(logits, dim=1)
            cur_labels = labels[mask]
            correct = th.sum(indices == cur_labels)
            return correct.item() * 1.0 / len(cur_labels)
    
    def make_graph(self, adj):
        g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph)
        g.remove_edges_from(nx.selfloop_edges(g))
        g = DGLGraph(g)
        g.add_edges(g.nodes(), g.nodes())
        if self.use_cuda:
            g = g.to('cuda:0')
        return g
        
    def forward(self, features, g):
        h = features
        layer_outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(g, h)
            layer_outputs.append(h)
        if self.operation == 'Maxpool':
            h = th.stack(layer_outputs, dim=0)
            h = th.max(h, dim=0)[0]
        else:
            h = th.cat(layer_outputs, dim=1)
        h = self.layers[-1](g, h)
        return F.log_softmax(h, dim=1)
    
    def one_step_training(self, features, g, labels,
                          idx_train, idx_val, idx_test,
                          lr, weight_decay, optimizer):
        self.train()
        logits = self.forward(features, g)
        loss = F.nll_loss(logits[idx_train], labels[idx_train])
        val_loss = F.nll_loss(logits[idx_val], labels[idx_val]).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = self.evaluate(features, g, labels, idx_train)
        val_acc = self.evaluate(features, g, labels, idx_val)
        test_acc = self.evaluate(features, g, labels, idx_test)
        return val_loss, [train_acc, val_acc, test_acc]
        
    
    def fit(self, adj, labels, features, idx_train, idx_val, idx_test,
            epochs=200 ,lr=5e-3, weight_decay=5e-4, patience=20, verbose=True):
        # First, let's process the data
        self.features = th.FloatTensor(features.todense())
        self.labels = th.LongTensor(labels)
        self.g = self.make_graph(adj)
        
        if self.use_cuda:
            self.cuda() # Model to cuda
            self.features = self.features.cuda() 
            self.labels = self.labels.cuda()
            # adj has been make cuda() in the make_graph function
            
        optimizer = th.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = np.inf
        selected_accs = None
        for epoch in range(1, epochs):
            if patience < 0:
                print("Early stopping happen at epoch %d." % epoch)
                break
                
            val_loss, accs = self.one_step_training(
                self.features, self.g, self.labels,
                idx_train, idx_val, idx_test,
                lr, weight_decay, optimizer)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                selected_accs = accs
                patience = patience
                if verbose:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, *accs))
            else: 
                patience -= 1
        log = 'Training finished. Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    
    def predict(self, adj=None, features=None):
        if adj is not None:
            self.g = self.make_graph(adj)
        if features is not None:
            self.features = th.FloatTensor(features.todense())
            if self.use_cuda:
                self.features = self.features.cuda() 
        self.eval()
        with th.no_grad():
            logits = self.forward(self.features, self.g)
            return logits

if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    citeseer_dataset = Dataset(root='../../datasets/data', name='citeseer') 
    print("================= Maxpool JKNet =================")
    JKNetModel = JKNet(in_feats=citeseer_dataset.features.shape[1],
                   out_feats=len(np.unique(citeseer_dataset.labels)), operation="Maxpool", use_cuda=True)
    JKNetModel.fit(adj=citeseer_dataset.adj, 
               features=citeseer_dataset.features,
               labels=citeseer_dataset.labels,
               idx_train=citeseer_dataset.idx_train,
               idx_test=citeseer_dataset.idx_test,
               idx_val=citeseer_dataset.idx_val)
    print(JKNetModel.predict())

    print("================= ConCat JKNet =================")
    JKNetModel = JKNet(in_feats=citeseer_dataset.features.shape[1],
                   out_feats=len(np.unique(citeseer_dataset.labels)), operation="ConCat", use_cuda=True)
    JKNetModel.fit(adj=citeseer_dataset.adj, 
               features=citeseer_dataset.features,
               labels=citeseer_dataset.labels,
               idx_train=citeseer_dataset.idx_train,
               idx_test=citeseer_dataset.idx_test,
               idx_val=citeseer_dataset.idx_val)
    print(JKNetModel.predict())

