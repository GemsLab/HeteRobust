import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import preprocess
from prognn_utils.PGD import *
import warnings

class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.
    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.
    """

    def __init__(self, nfeat, nclass, nhid=16, dropout=0.5, debug=False, cuda_=False):
        self.device = 'cuda' if cuda_ else 'cpu'
        
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.dropout=dropout
        self.debug = debug
        
        # Initiate the GCN model
        self.model = GCN(nfeat=nfeat,
                         nhid=nhid,
                         nclass=nclass,
                         dropout=dropout, 
                         device=self.device)
        self.model = self.model.to(self.device)

    def fit(self, features, adj, labels, idx_train, idx_val,
            lr=0.01, lr_adj=0.01, weight_decay=5e-4, symmetric=False,
            alpha=5e-4, beta=1.5, gamma=1, lambda_=0, phi=0,
            epochs=400, only_gcn=False, outer_steps=1, inner_steps=2):
        """Train Pro-GNN.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.lr = lr
        self.lr_adj = lr_adj
        self.weight_decay = weight_decay
        self.symmetric = symmetric
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.phi = phi
        self.epochs = epochs
        self.only_gcn = only_gcn
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        
        # Perform the preprocessing of the input
        self.labels_dummy = labels
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=self.device)
        self.features = features
        self.adj = adj
        self.labels = labels
        

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        estimator = EstimateAdj(adj, symmetric=self.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=self.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=self.lr_adj, alphas=[self.alpha])

        # warnings.warn("If you find the nuclear proximal operator runs too slow on Pubmed, you can  uncomment line 67-71 and use prox_nuclear_cuda to perform the proximal on gpu.")
        # if self.dataset == "pubmed":
        #     self.optimizer_nuclear = PGD(estimator.parameters(),
        #               proxs=[prox_operators.prox_nuclear_cuda],
        #               lr=self.lr_adj, alphas=[self.beta])
        # else:
#         warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        if self.device=='cpu':
            self.optimizer_nuclear = PGD(estimator.parameters(),
                      proxs=[prox_operators.prox_nuclear],
                      lr=self.lr_adj, alphas=[self.beta])
        else:
            self.optimizer_nuclear = PGD(estimator.parameters(),
                      proxs=[prox_operators.prox_nuclear_cuda],
                      lr=self.lr_adj, alphas=[self.beta])

        # Train model
        t_total = time.time()
        for epoch in range(self.epochs):
            if self.only_gcn:
                self.train_gcn(epoch, features, estimator.estimated_adj,
                        labels, idx_train, idx_val)
            else:
                for i in range(int(self.outer_steps)):
                    self.train_adj(epoch, features, adj, labels,
                            idx_train, idx_val)

                for i in range(int(self.inner_steps)):
                    self.train_gcn(epoch, features, estimator.estimated_adj,
                            labels, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if self.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))



    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        if self.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + self.gamma * loss_gcn + self.lambda_ * loss_smooth_feat + self.phi * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if self.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                    + self.gamma * loss_gcn \
                    + self.alpha * loss_l1 \
                    + self.beta * loss_nuclear \
                    + self.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if self.debug:
            print('Epoch: {:04d}'.format(epoch+1),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if self.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))
    
    def eval(self):
        self.model.eval()
    
    def predict(self, adj=None, features=None):
        if adj is not None or features is not None:
            adj, features, _ = preprocess(adj, features, self.labels_dummy, 
                                          preprocess_adj=False, device=self.device)
            self.adj = adj
            self.features = features
        # Question: can we do evasion setting here?
        self.model.eval()
        return self.model(self.features, self.adj)
        
    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        labels = self.labels
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


if __name__ == '__main__':
    from deeprobust.graph.data import Dataset
    import pickle
    cur_dataset = Dataset(root='../../datasets/data', name='cora')
    with open("../../workspace/372312abe142169dcc816a232722add2/perturbDict.pkl", 'rb') as f:
        perturb_dict = pickle.load(f)
    features = cur_dataset.features
    adj = cur_dataset.adj
    PRED_NAME = 'p_m0_174'
    perturb_dict = perturb_dict[PRED_NAME]
    features = features + perturb_dict['features']
    adj = adj + perturb_dict['adj']

    # TODO: make sure that the arguments align with the nettack session
    flag_gpu = torch.cuda.is_available()
    print("Using gpu:", flag_gpu)
    ProGNN_model = ProGNN(nfeat=features.shape[1], 
        nclass=cur_dataset.labels.max()+1, 
        cuda_=flag_gpu,
        nhid=64)

    ProGNN_model.fit(features, 
              adj, 
              cur_dataset.labels,
              cur_dataset.idx_train, 
              cur_dataset.idx_val,
              epochs=200)
    ProGNN_model.test(cur_dataset.idx_test)
    print(ProGNN_model.predict())
    
