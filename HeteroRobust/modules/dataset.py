from deeprobust.graph.data import Dataset
import os.path as osp
import numpy as np
import scipy.sparse as sp

class CustomDataset(Dataset):
    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!"
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test, self.idx_wild = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True, is_sparse_attr=None):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse_attr is None:
                is_sparse_attr = loader.get("is_sparse_attr", is_sparse)

            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                labels = loader.get('labels')

            if 'attr_data' in loader:
                if is_sparse_attr:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = loader['attr_data']
            else:
                features = None
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
                idx_wild = loader.get("idx_wild")
                if idx_wild is None:
                    idx_wild = []
        else:
            idx_train, idx_val, idx_test = super().get_train_val_test()
            idx_wild = []

        return idx_train, idx_val, idx_test, idx_wild


class DataPreprocess:
    @staticmethod
    def gdc(A: sp.csr_matrix, alpha: float, eps: float):
        N = A.shape[0]
        # Self-loops
        A_loop = sp.eye(N) + A
        # Symmetric transition matrix
        D_loop_vec = A_loop.sum(0).A1
        D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
        D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
        T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

        # PPR-based diffusion
        S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
        # Sparsify using threshold epsilon
        S_tilde = S.multiply(S >= eps)

        # Column-normalized transition matrix on graph S_tilde
        D_tilde_vec = S_tilde.sum(0).A1
        T_S = S_tilde / D_tilde_vec
        return T_S
