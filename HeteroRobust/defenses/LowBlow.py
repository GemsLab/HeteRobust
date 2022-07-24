import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np
from numba import njit
from deeprobust.graph import utils

def normalize_adj_tensor(adj, sparse=False, add_eye=True):
    """Normalize adjacency tensor matrix.
    """
    device = torch.device("cuda" if adj.is_cuda else "cpu")
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = utils.to_scipy(adj)
        mx = normalize_adj(adj, add_eye)
        return utils.sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        if add_eye:
            mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

def normalize_adj(mx, add_eye=True):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 and add_eye:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


class GCNSVD(GCN):
    """GCNSVD is a 2 Layer Graph Convolutional Network with Truncated SVD as
    preprocessing. See more details in All You Need Is Low (Rank): Defending
    Against Adversarial Attacks on Graphs,
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
	We can first load dataset and then train GCNSVD.
    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNSVD
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNSVD(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=20)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, k=50,
                 weight_decay=5e-4, with_relu=True, with_bias=True, svd_solver="default", threshold=0.1):
        super(GCNSVD, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, 
        	device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) #pylint: disable=no-member
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
        self.k = k
        assert svd_solver in ["default", "authors", "eigen", "eye-svd"]
        self.svd_solver = svd_solver
        self.threshold = threshold # Only used if svd_solver == "authors"

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=True, **kwargs):
        """First perform rank-k approximation of adjacency matrix via
        truncated SVD, and then train the gcn model on the processed graph,
        when idx_val is not None, pick the best model according to
        the validation loss.
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
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        k : int
            number of singular values and vectors to compute.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        self.labels = labels
        if self.svd_solver == "eye-svd":
            normalize = False
            if sp.issparse(adj):
                if type(adj) is not sp.lil.lil_matrix:
                    adj = adj.tolil()
                adj.setdiag(1)
                modified_adj = self.truncatedSVD(adj, k=self.k)
                modified_adj[modified_adj < 0] = 0 # This is necessary to avoid loss becoming nan
                features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
                modified_adj = normalize_adj_tensor(modified_adj, sparse=True, add_eye=False)
            else:
                mx = adj + torch.eye(adj.shape[0]).to(self.device)
                modified_adj = self.truncatedSVD(mx, k=self.k)
                modified_adj[modified_adj < 0] = 0 # This is necessary to avoid loss becoming nan
                features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
                modified_adj = normalize_adj_tensor(modified_adj, sparse=False, add_eye=False)
        else:
            normalize = True
            modified_adj = self.truncatedSVD(adj, k=self.k)
            # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
            features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels

        super().fit(features, modified_adj, labels, idx_train, idx_val,
                    train_iters=train_iters, initialize=initialize, verbose=verbose, 
                    normalize=normalize)
        
    
    def truncatedSVD(self, data, k=50):
        
        """Truncated SVD on input data.
        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.
        Returns
        -------
        numpy.array
            reconstructed matrix.
        """

        print('=== GCN-SVD: rank={} ==='.format(k))
        if self.svd_solver in ["default", "authors", "eye-svd"]:
            if sp.issparse(data):
                data = data.asfptype()
                U, S, V = sp.linalg.svds(data, k=k)
                print("rank_after = {}".format(len(S.nonzero()[0])))
                diag_S = np.diag(S)
            else:
                U, S, V = np.linalg.svd(data)
                U = U[:, :k]
                S = S[:k]
                V = V[:k, :]
                diag_S = np.diag(S)
            lr_adj = U @ diag_S @ V
            if self.svd_solver == "authors":
                return (lr_adj > self.threshold).astype(V.dtype)
            else:
                return lr_adj

        elif self.svd_solver == "eigen":
            # scipy.sparse.linalg.eigsh can be used
            # If k is positive, use which="LA" mode to get top-k largest eigenvalues > 0
            # If k is negative, use which="SA" mode to get top-k largest (in magnitude) eigenvalues < 0
            print("### Using EigenDecomposition ###")
            if sp.issparse(data):
                data = data.asfptype()
                # Make sure this is a symmetric matrix
                assert np.sum(data-data.T) == 0.0
                if k > 0:
                    eigenvalues, eigenvectors = eigsh(data, k=k, which="LA")
                elif k < 0:
                    eigenvalues, eigenvectors = eigsh(data, k=-k, which="SA")
                else:
                    raise Exception("k should not be zero")
                # Since input is symmetric, use eigenvector.T
                return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            else:
                raise NotImplementedError(
                    "Eigendecomposition for GCNSVD has not been implemented for dense adjacency matrix.")
    
    def predict(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            modified_adj = self.truncatedSVD(adj, k=self.k)
            features, modified_adj, labels = utils.to_tensor(features, modified_adj, self.labels, device=self.device)
            if utils.is_sparse_tensor(modified_adj):
                modified_adj = utils.normalize_adj_tensor(modified_adj, sparse=True)
            else:
                modified_adj = utils.normalize_adj_tensor(modified_adj)
            
            return self.forward(features, modified_adj)

if __name__ == '__main__':
	from deeprobust.graph.data import Dataset
	import torch
	citeseer_dataset = Dataset(root='../../datasets/data', name='citeseer') 
	model = GCNSVD(nfeat=citeseer_dataset.features.shape[1], 
               nclass=citeseer_dataset.labels.max()+1,
                nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4, 
                k=50, # Number of singular values and vectors to compute
                svd_solver='eye-svd'
                )
	model.fit(features=citeseer_dataset.features,
          adj=citeseer_dataset.adj,
          labels=citeseer_dataset.labels,
          idx_train=citeseer_dataset.idx_train,
          idx_val=citeseer_dataset.idx_val,
          train_iters=100,
          verbose=True
          )
	print(model.predict())
	print(model.predict(features=citeseer_dataset.features,
              adj=citeseer_dataset.adj))
