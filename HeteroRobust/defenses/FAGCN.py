from .FAGCN_utils.model import FAGCN_base
from .FAGCN_utils.utils import *
import torch
import types
from deeprobust.graph import utils
import dgl
from dgl import DGLGraph

class FAGCN():
    def __init__(self, dataset, nfeat, nclass, dropout=0.2, nhid=64,
                       lr=0.01, depth=2, eps=0.1, weight_decay=5e-4):

        self.device = torch.device('cpu')

        if type(dataset.adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(dataset.features, dataset.adj, dataset.labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        self.adj = adj._indices()
        self.num_nodes = adj.shape[0]
        self.labels = labels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = self.reformat_adj(self.adj, self.num_nodes)
        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.labels = labels.to(self.device)
        self.g = g.to(self.device)

        # Args we need to expose to adversarial attack framework
        self.nfeat = nfeat
        self.nclass = nclass
        self.hidden_sizes = [nhid]
        self.with_relu = True
        self.with_bias = True
        self.output = None

        self.model = FAGCN_base(g=self.g, in_dim=nfeat, hidden_dim=nhid, out_dim=nclass,
                                dropout=dropout, eps=eps, layer_num=depth, lr=lr, wd=weight_decay)
        self.model.to(self.device)


    def reformat_adj(self, adj, num_nodes):
        g = dgl.graph((adj[0, :], adj[1, :]), num_nodes=num_nodes)
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        g = g.to(self.device)
        deg = g.in_degrees().to(self.device).float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm

        return g

    def fit(self, features, adj, labels, idx_train, idx_val,
                  idx_test=None, train_iters=100, verbose=True):

        self.device = torch.device('cpu')
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = features.to_dense()
        self.features = normalize_features(self.features)
        self.features = torch.FloatTensor(self.features).to(self.device)
        self.adj = adj._indices()
        self.num_nodes = adj.shape[0]
        self.labels = labels
        self.labels = torch.LongTensor(labels).to(self.device)
        g = self.reformat_adj(self.adj, self.num_nodes)
        self.g = g.to(self.device)
        
        self.model.train_model(self.g, self.features, self.labels, idx_train, idx_val, idx_test, train_iters)

        return None


    def predict(self, features=None, adj=None, idx=None):


        self.device = torch.device('cpu')
        if features is not None or adj is not None:
            features, adj = utils.to_tensor(features, adj, device=self.device)
            features = features.to_dense()
            features = normalize_features(features)
            self.features = torch.Tensor(features)
            self.adj = adj._indices()
            self.num_nodes = adj.shape[0]
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.g = self.reformat_adj(self.adj, self.num_nodes).to(self.device)
        self.model.g = self.g
        self.model.reset_g()
        self.features = self.features.to(self.device)
        out = self.model.forward(self.features)
        if idx is not None:
            out = out[idx, :]
        self.output = out
        return out

    def eval(self):
        self.model.eval()


if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    from ..modules.dataset import CustomDataset

    citeseer_dataset = CustomDataset(root='datasets/data', name='citeseer')
    FAGCN_model = FAGCN(dataset=citeseer_dataset, nfeat=citeseer_dataset.features.shape[1], \
                        nclass=citeseer_dataset.labels.max().item()+1)

    FAGCN_model.fit(features=citeseer_dataset.features,
                    adj=citeseer_dataset.adj,
                    labels=citeseer_dataset.labels,
                    idx_train=citeseer_dataset.idx_train,
                    idx_val=citeseer_dataset.idx_val,
                    idx_test=citeseer_dataset.idx_test,
                    train_iters=200)
    FAGCN_model.eval()
    print(FAGCN_model.predict(idx=citeseer_dataset.idx_test))
