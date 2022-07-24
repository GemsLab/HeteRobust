import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from .gat_utils.gat_conv import GATConv
from torch_geometric.data import InMemoryDataset, Data
from itertools import repeat

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

class Dpr2Pyg(InMemoryDataset):
    """Convert deeprobust data (sparse matrix) to pytorch geometric data (tensor, edge_index)
    Parameters
    ----------
    dpr_data :
        data instance of class from deeprobust.graph.data, e.g., deeprobust.graph.data.Dataset,
        deeprobust.graph.data.PtbDataset, deeprobust.graph.data.PrePtbDataset
    transform :
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before every access. For example, you can
        use torch_geometric.transforms.NormalizeFeatures()
    Examples
    --------
    We can first create an instance of the Dataset class and convert it to
    pytorch geometric data format.
    >>> from deeprobust.graph.data import Dataset, Dpr2Pyg
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> pyg_data = Dpr2Pyg(data)
    >>> print(pyg_data)
    >>> print(pyg_data[0])
    """

    def __init__(self, dpr_data, transform=None, **kwargs):
        self.transform = transform
        pyg_data = self.process(dpr_data)
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform
        self.__indices__ = None

    def process(self, dpr_data):
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = index_to_mask(idx_train, size=y.size(0))
        val_mask = index_to_mask(idx_val, size=y.size(0))
        test_mask = index_to_mask(idx_test, size=y.size(0))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def update_edge_index(self, adj):
        self.data.edge_index = torch.LongTensor(adj.nonzero())
        self.data, self.slices = self.collate([self.data])

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, dpr_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
   
        if initialize:
            self.initialize()

        self.data = Dpr2Pyg(dpr_data)[0].to(self.device)
        self.dpr_data = dpr_data
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)
    
    def predict(self, adj=None, features=None):
        self.eval()
        if adj is None and features is None:
            return self.forward(self.data)
        
        dpr_data = deepcopy(self.dpr_data)
        dpr_data.adj = adj
        dpr_data.features = features
        return self.forward(Dpr2Pyg(dpr_data)[0].to(self.device))
        
    def test(self):
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

if __name__ == '__main__':
	from deeprobust.graph.data import Dataset
	citeseer_dataset = Dataset(root='datasets/data', name='citeseer') 
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	gat_model = GAT(nfeat=citeseer_dataset.features.shape[1],
                nhid=8,
                heads=8,
                nclass=citeseer_dataset.labels.max()+1,
                dropout=0.5,
                device=device
               )
	gat_model.fit(citeseer_dataset, verbose=True)
	print(gat_model.predict(adj=citeseer_dataset.adj, features=citeseer_dataset.features))