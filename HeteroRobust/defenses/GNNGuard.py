from .GNNGuard_utils.gcn_gnnguard import GCN
from .GNNGuard_utils.jumpingknowledge import JK
from .GNNGuard_utils.gcn_gnnguard_fixed import GCN as GCN_fixed
import torch

class GNNGuard():
    def __init__(self, nfeat, nclass, dropout=0.5, nhid=256, 
                        lr=0.01, weight_decay=5e-4, n_edge=1, # Specific to JKNet
                        base_model='GCN'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        if self.base_model == 'GCN':
            self.underlying_model = GCN(
                nfeat=nfeat, 
                nhid=nhid, 
                nclass=nclass,
                dropout=dropout,
                device=self.device)
        elif self.base_model == 'JK':
            self.underlying_model = JK(
                nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                n_edge=n_edge,
                device=self.device
            )
        elif self.base_model == 'GCN_fixed':
            self.underlying_model = GCN_fixed(
                nfeat=nfeat, 
                nhid=nhid, 
                nclass=nclass,
                dropout=dropout,
                device=self.device)
        else:
            print("Error: Please choose from ['GCN', 'JKNet']")
        self.underlying_model.to(self.device)
    
    def fit(self, features, adj, labels, idx_train, idx_val, idx_test, train_iters=81, verbose=True,
                    att_0=None, attention=False, normalize=False, patience=500):
        if self.base_model in ['GCN', 'GCN_fixed']:
            self.underlying_model.fit(features, adj, labels,
                                    idx_train, idx_val=idx_val, idx_test=idx_test,
                                    attention=True, verbose=verbose, train_iters=train_iters)
        elif self.base_model == 'JK':
            self.underlying_model.fit(features, adj, labels, 
                                    idx_train, idx_val, idx_test,
                                    train_iters=train_iters, attention=attention, patience=patience)
    
    def eval(self):
        self.underlying_model.eval()
    
    def predict(self, features=None, adj=None):
        return self.underlying_model.predict(features=features, adj=adj)


if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    from ..modules.dataset import CustomDataset
    citeseer_dataset = CustomDataset(root='datasets/data', name='FB100') 
    print(citeseer_dataset.features.shape[1])
    GNNGuard_model = GNNGuard(nfeat=citeseer_dataset.features.shape[1],
                          nclass=citeseer_dataset.labels.max().item()+1,
                          base_model='GCN_fixed')
    GNNGuard_model.fit(features=citeseer_dataset.features,
                     adj=citeseer_dataset.adj,
                     labels=citeseer_dataset.labels,
                     idx_train=citeseer_dataset.idx_train,
                     idx_val=citeseer_dataset.idx_val,
                     idx_test=citeseer_dataset.idx_test,
                     train_iters=5)
    GNNGuard_model.eval()
    print(GNNGuard_model.predict(citeseer_dataset.features, citeseer_dataset.adj))
    print(GNNGuard_model.underlying_model.test(citeseer_dataset.idx_test)[0])
