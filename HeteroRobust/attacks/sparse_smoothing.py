import numpy as np
import pickle
from numpy.core.records import record
import pandas
import re
import signac
import torch
import torch.nn.functional as F

from pathlib import Path
from scipy.sparse import csr_matrix
from .base import AttackSession, recordcalls
from .modules.sparse_smoothing.prediction import predict_smooth_gnn
from .modules.sparse_smoothing.cert import p_lower_from_votes, binary_certificate_grid, joint_binary_certificate_grid

class SparseSmoothingSession(AttackSession):
    @recordcalls
    def __init__(self, datasetName="citeseer", random_seed=15, debug=False,
                 n_samples=100_000, pf_plus_adj=0.001, pf_minus_adj=0.4, 
                 pf_plus_att=0.0, pf_minus_att=0.0, batch_size=1,
                 conf_alpha=0.01):
        super().__init__(datasetName=datasetName, random_seed=random_seed, debug=debug)

        self.edge_idx = torch.LongTensor(self.data.adj.nonzero())
        self.attr_idx = torch.LongTensor(self.data.features.nonzero())
        self.num_nodes, self.feature_dim = self.data.features.shape
        self.num_classes = self.data.labels.max().item() + 1

        if self.cuda:
            self.edge_idx = self.edge_idx.cuda()
            self.attr_idx = self.attr_idx.cuda()

        self.sample_config = {
            'n_samples': n_samples,
            'pf_plus_adj': pf_plus_adj,
            'pf_minus_adj': pf_minus_adj,
            'pf_plus_att': pf_plus_att,
            'pf_minus_att': pf_minus_att
        }
        self.batch_size = batch_size
        self.conf_alpha = conf_alpha

        self.votesDict = dict()
        self.sampleConfigDict = dict()
        self.pValuesDict = dict()
        self.certGridDict = dict()
        self.regionDict = dict()
        
    @recordcalls
    def predict_smooth_gnn(self, defense_model:str, target="votes", **kwargs):
        _model = self.modelDict[defense_model]
        sample_config = self.sample_config.copy()

        for key, value in kwargs.items():
            if key in self.sample_config:
                sample_config[key] = value
        
        self.sampleConfigDict[target] = sample_config
        self.votesDict[target] = predict_smooth_gnn(
            attr_idx=self.attr_idx, edge_idx=self.edge_idx,
            sample_config=sample_config,
            model=_model, n=self.num_nodes, d=self.feature_dim, 
            nc=self.num_classes, batch_size=self.batch_size
        )
        
        return self
    
    @recordcalls
    def certificate_grid(self, prevotes_target="prevotes", votes_target="votes", conf_alpha=None):
        if conf_alpha is None:
            conf_alpha = self.conf_alpha

        self._noTrace = True
        statePointDict = dict(
            type="certificate",
            certModel="sparse_smoothing",
            datasetName=self.datasetName,
            random_seed=self.random_seed,
            sampleConfig=dict(
                preVotes=self.sampleConfigDict[prevotes_target],
                votes=self.sampleConfigDict[votes_target]
            ),
            conf_alpha=conf_alpha
        )
        self.signacJob = self.create_signac_job(statePointDict, sp_include_init=False)

        with self.signacJob.data:
            self.signacJob.data.votes = self.votesDict[votes_target]
            self.signacJob.data.preVotes = self.votesDict[prevotes_target]
        self.signacJob.doc.voteSaved = True

        p_lower = p_lower_from_votes(
            votes=self.votesDict[votes_target],
            pre_votes=self.votesDict[prevotes_target],
            alpha = conf_alpha,
            n_samples=self.sampleConfigDict[votes_target]["n_samples"]
        )
        p_value_target = votes_target + "_p_lower"
        self.pValuesDict[p_value_target] = p_lower
        with self.signacJob.data:
            self.signacJob.data.pValues = dict(p_value_target=p_lower)
        self.signacJob.doc.pValuesSaved = True
        print(f"Saved p_lower values to signac job {self.signacJob.id}")

        if (self.sample_config["pf_plus_adj"] == 0) and (self.sample_config["pf_minus_adj"] == 0):
            pf_plus = self.sample_config["pf_plus_att"]
            pf_minus = self.sample_config["pf_minus_att"]
        elif (self.sample_config["pf_plus_att"] == 0) and (self.sample_config["pf_minus_att"] == 0):
            pf_plus = self.sample_config["pf_plus_adj"]
            pf_minus = self.sample_config["pf_minus_adj"]
        else:
            pf_plus = None
        
        if pf_plus is None:
            raise NotImplementedError()
            joint_binary_certificate_grid(
                pf_plus_adj=self.sample_config["pf_plus_adj"],
                pf_minus_adj = self.sample_config["pf_minus_adj"],
                pf_plus_att = self.sample_config["pf_plus_att"],
                pf_minus_att = self.sample_config["pf_minus_att"],
                p_emps=p_lower, reverse=False, progress_bar=True
            )

            self.certGridDict[votes_target] = dict()
            self.certGridDict[votes_target]["type"] = "joint"
        else:
            rho_grid, regions, max_ra, max_rd = binary_certificate_grid(
                pf_plus=pf_plus, pf_minus=pf_minus,
                p_emps=p_lower, reverse=False, progress_bar=True
            )
            self.certGridDict[votes_target] = dict()
            self.certGridDict[votes_target]["type"] = "separate"
            self.certGridDict[votes_target]["rho_grid"] = rho_grid
            self.certGridDict[votes_target]["max_ra"] = max_ra
            self.certGridDict[votes_target]["max_rd"] = max_rd

            self.regionDict[votes_target] = regions
        
        with self.signacJob.data:
            self.signacJob.data.certGrid = self.certGridDict[votes_target]
        self.signacJob.doc.certGridSaved = True
        print(f"Saved certification grid to signac job {self.signacJob.id}")

        with self.signacJob:
            with open("regionDict.pkl", "wb") as f:
                pickle.dump(self.regionDict, f)
            with open("data.pkl", "wb") as f:
                pickle.dump(self.data, f)
            
        if not self.debug:
            self.signacJob.doc.success = True

        self._noTrace = False
        return self