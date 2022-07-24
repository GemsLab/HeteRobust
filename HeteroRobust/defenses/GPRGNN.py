from .GPRGNN_utils.GNN_models import GPRGNN_base
import torch
import types
from deeprobust.graph import utils

class GPRGNN():
    def __init__(self, nfeat, nclass, dropout=0.2, nhid=64,
                       lr=0.01, Init='PPR', K=10, alpha=0.1,
                       dprate=0.2, weight_decay=5e-4, Gamma=None, 
                       bias=True):

        # Args we need to expose to model
        args = types.SimpleNamespace()
        args.num_features = nfeat
        args.num_class = nclass
        args.hidden = nhid
        args.ppnp = 'GPR_prop'
        args.K = K
        args.alpha = alpha
        args.Init = Init
        args.Gamma = Gamma
        args.dprate = dprate
        args.dropout = dropout 
        args.bias = bias 
        args.lr = lr
        args.weight_decay = weight_decay

        # Used for simple prediction calls
        self.features = None
        self.adj = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Args we need to expose to adversarial attack framework
        self.nfeat = nfeat
        self.nclass = nclass
        self.hidden_sizes = [nhid]
        self.with_relu = True 
        self.with_bias = True
        self.output = None 

        self.model = GPRGNN_base(args)
        self.model.to(self.device) 


    # Fits base model with given params, also saves data to model object 
    def fit(self, features, adj, labels, idx_train, idx_val, 
                  idx_test=None, train_iters=100, verbose=True):

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

        self.model.train_model(data, idx_train, idx_val, idx_test, train_iters)
        
        return None

 
    # Runs prediction on current model, populates output variable 
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

        out = self.model.forward(data) 
        if idx is not None:
            out = out[idx, :]
        self.output = out
        return out

    def eval(self):
        self.model.eval()
 

if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    from ..modules.dataset import CustomDataset
    
    citeseer_dataset = CustomDataset(root='datasets/data', name='FB100')

    print(citeseer_dataset.features.shape[1])
    GPRGNN_model = GPRGNN(nfeat=citeseer_dataset.features.shape[1], \
                                  nclass=citeseer_dataset.labels.max().item()+1) 

    GPRGNN_model.fit(features=citeseer_dataset.features,
                     adj=citeseer_dataset.adj,
                     labels=citeseer_dataset.labels,
                     idx_train=citeseer_dataset.idx_train,
                     idx_val=citeseer_dataset.idx_val,
                     idx_test=citeseer_dataset.idx_test,
                     train_iters=200)

    GPRGNN_model.eval()
    print(GPRGNN_model.predict(idx=citeseer_dataset.idx_test))
