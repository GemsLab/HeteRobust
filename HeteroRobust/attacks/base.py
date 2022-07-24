import numpy as np
import torch
import fire
import builtins
import pandas
import re
import itertools
import signac
import pickle
import os.path as osp
import warnings

from enum import Enum
from inspect import signature
from functools import wraps
from pathlib import Path
from datetime import datetime
from ..modules.dataset import CustomDataset, DataPreprocess
from deeprobust.graph.data import Dataset
from copy import copy, deepcopy
from deeprobust.graph.defense import GCN
from ..defenses.GCN_multi import MultiLayerGCN
from ..defenses.H2GCN import SubprocessModel, H2GCN, CPGNN
from ..defenses.graphsage_simple import GraphSage
from ..defenses.JKNet import JKNet
from ..defenses.prognn import ProGNN
from ..defenses.GNNGuard import GNNGuard
from ..defenses.GAT_deeprobust import GAT
from ..defenses.LowBlow import GCNSVD
from ..defenses.GPRGNN import GPRGNN
from ..defenses.FAGCN import FAGCN
from ..defenses.APPNP import APPNP

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


def recordcalls(func):
    sig = signature(func)

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not getattr(obj, "_noTrace", False):
            s = sig.bind(obj, *args, **kwargs)
            s.apply_defaults()
            argumentsDict = s.arguments
            argumentsDict.pop("self", False)
            argumentsDict.pop("help", False)
            argumentsDict.pop("debug", False)
            if not hasattr(obj, "sessionTrace"):
                obj.sessionTrace = dict()
            if func.__name__ == "__init__":
                key = f"{func.__name__}"
            else:
                key = f"{func.__name__}_{wrapper.calls}"
                while key in obj.sessionTrace:
                    wrapper.calls += 1
                    key = f"{func.__name__}_{wrapper.calls}"
            obj.sessionTrace[key] = argumentsDict
            wrapper.calls += 1
        return func(obj, *args, **kwargs)
    wrapper.calls = 0
    return wrapper


class AttackSession:
    @staticmethod
    def _debug():
        try:
            import ptvsd
            print("Waiting for debugger attach. Press CTRL+C to skip.")
            ptvsd.enable_attach(address=('localhost', 5678),
                                redirect_output=True)
            ptvsd.wait_for_attach()
            breakpoint()
        except KeyboardInterrupt:
            pass

    def __init__(self, datasetName="citeseer",
                 random_seed=15, debug=False):
        # TODO: allow CUDA to be disabled...
        if debug == "vs":
            self._debug()
            self.debug = True
        else:
            self.debug = debug
        self.cuda = torch.cuda.is_available()
        print('cuda: %s' % self.cuda)
        self.device = torch.device(  # pylint: disable=no-member
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.cuda:
            torch.cuda.manual_seed(self.random_seed)
        self.datasetName = datasetName
        # Newly added to avoid error when resume from session
        re_syn = None
        if datasetName is not None:
            re_syn = re.match(r"(syn-\w*):([^:]+)(?::(\w+))?", datasetName)
            if re_syn:
                data_folder = re_syn.group(1)
                filename = re_syn.group(2)
                setting = re_syn.group(3)
                if setting is None:
                    setting = "nettack"
                print(f"Using {setting} setting for training, validation and testing splits.")
                self.data = CustomDataset(root=f"datasets/{data_folder}", name=filename, setting=setting)
                self.data.__class__ = Dataset  # Cast the child class back to parent class
            else:
                if datasetName in ['citeseer', 'cora', 'cora_ml', 'pubmed']:
                    self.data = Dataset(root='datasets/data', name=datasetName)
                elif datasetName in ['deezer-europe', 'twitch-explicit', 'snap-patents-downsampled', 'pokec', 'fb100', 'twitch-tw']:
                    root = osp.expanduser(osp.normpath('datasets/data'))
                    data_folder = osp.join(root, datasetName)
                    data_filename = data_folder + '.pkl'
                    # Make sure dataset file exists
                    assert osp.exists(data_filename), f"{data_filename} does not exist!"
                    with open (data_filename, 'rb') as f:
                        print(data_filename)
                        self.data = CustomUnpickler(f).load()
                        print(f"Successfully loaded {datasetName} dataset!")
                        self.data.__class__ = Dataset  # Cast the child class back to parent class
                else:
                    raise Exception(f"Unknown dataset {datasetName}.")

                # adj, features, labels = data.adj, data.features, data.labels
                # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
                # idx_unlabeled = np.union1d(idx_val, idx_test)
            if self.data.adj.diagonal().any():
                if datasetName in ['pubmed']:
                    print("Removing self loops in adjacency matrix")
                    self.data.adj = self.data.adj.tolil()
                    self.data.adj.setdiag(0)
                    self.data.adj = self.data.adj.astype("float32").tocsr()
                    self.data.adj.eliminate_zeros()
                else:
                    raise RuntimeError("This dataset has an adjacency matrix with non-empty diagonal elements. "
                                   "Please remove diagonal elements to avoid potential issues in preprocessing.")

        self.modelDict = dict()
        self.perturbDict = dict()
        self.predictionDict = dict()
        self.signacJob = None
        self.signacJobPerturb = None
        self.signacJobHist = []
        self.overwriteJob = False
        self.use_runner = False

    @recordcalls
    def add_model(self, model_type: str, name=None, help=False, from_json=None, **kwargs):
        if name is None:
            name = "m" + str(len(self.modelDict))

        model_class = eval(model_type)
        if help:
            builtins.help(model_class)
            exit()

        if model_class in [GCN]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                nhid=16, dropout=0.5, with_relu=True, with_bias=False,
                device=self.device
            )
        elif model_class in [MultiLayerGCN]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                nhid=16, nlayer=2, dropout=0.5, with_relu=True, with_bias=False,
                device=self.device
            )
        elif model_class in [H2GCN, CPGNN]:
            paramsDict = dict(
                random_seed=self.random_seed,
                verbose=self.debug
            )
        elif model_class in [GraphSage]:
            paramsDict = dict(
                num_classes=self.data.labels.max().item()+1,
                model_class="SupervisedGraphSage",
                hid_units=128,
                cuda_=self.cuda,
                num_samples=[None, None]
            )
        elif model_class in [JKNet]:
            paramsDict = dict(
                in_feats=self.data.features.shape[1],
                out_feats=self.data.labels.max().item()+1,
                n_layers=6,
                n_units=32,
                dropout=0.5,
                operation="ConCat",
                use_cuda=True,
            )
        elif model_class in [ProGNN]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                nhid=16, dropout=0.5, cuda_=True
            )
        elif model_class in [GNNGuard]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                dropout=0.5,
                nhid=256,
                base_model='JK'  # Choose from 'JK' or 'GCN'
            )
        elif model_class in [GAT]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nhid=8, heads=8,
                nclass=self.data.labels.max().item()+1,
                dropout=0.5,
                device='cuda'
            )
        elif model_class in [GCNSVD]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                nhid=16, dropout=0.5, lr=0.01, weight_decay=5e-4,
                k=50  # Number of singular values and vectors to compute
            )
        elif model_class in [GPRGNN]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1,
                nhid=64, dropout=0.5, lr=0.01, weight_decay=5e-4,
                K=10, alpha=0.1, Init='PPR'
            )
        elif model_class in [FAGCN]:
            paramsDict = dict(
                dataset=self.data,
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1, eps=0.3,
                nhid=64, dropout=0.5, lr=0.01, depth=2, weight_decay=5e-4,
            )
        elif model_class in [APPNP]:
            paramsDict = dict(
                nfeat=self.data.features.shape[1],
                nclass=self.data.labels.max().item()+1, alpha=.1, K=2,
                nhid=64, dropout=0.5, lr=0.01, weight_decay=5e-4,
            )
        if from_json:
            raise NotImplementedError()
        paramsDict.update(kwargs)
        self.modelDict[name] = model_class(**paramsDict)
        if model_class in [GCN]:
            self.modelDict[name].to(self.device)
        print(f"<=== Added Model {name} \n {self.modelDict[name]} \n")
        return self

    @recordcalls
    def fit_models(self, model_names, data_name="clean", initialize=True,
                   save_predictions=True, preprocess=None, **kwargs):
        data = self.getData(data_name)
        adj_preprocess_flags = set()
        for name, model in self.get_models(model_names).items():
            if preprocess is not None:
                preprocess = preprocess.lower()
                # GDC
                preRe = re.match(r"gdc(?:-a([\d\.]+))?(?:-e([\d\.]+))?$", preprocess)
                if preRe:
                    if type(model) not in [H2GCN, CPGNN]:
                        if f"clean_{preprocess}" in self.perturbDict:
                            data = self.getData(f"clean_{preprocess}")
                        else:
                            print("Preprocessing dataset with GDC...")
                            alpha = float(preRe.group(1)) if preRe.group(1) else 0.1
                            eps = float(preRe.group(2)) if preRe.group(2) else 0.0005
                            data.adj = DataPreprocess.gdc(self.data.adj, alpha=alpha, eps=eps)
                            adj_preprocess_flags.add("gdc")
                            self.perturbDict[f"clean_{preprocess}"] = deepcopy(data)
                            if type(model) not in [GCN, MultiLayerGCN]:
                                warnings.warn("GDC preprocess is only tested with GCN.")
                    else:
                        kwargs["adj_norm_type"] = preprocess
                
                # If nothing matches
                if preRe is None:
                    raise ValueError(f"Unknown dataset preprocess setting {preprocess}!")
                    
            if type(model) in [GCN]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    patience=30,
                    normalize=(False if (set(["gdc"])
                        & adj_preprocess_flags) else True),
                    verbose=self.debug
                )
            elif type(model) in [MultiLayerGCN]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    patience=30,
                    normalize=(False if (set(["gdc"])
                        & adj_preprocess_flags) else True),
                    verbose=self.debug
                )
            elif type(model) in [H2GCN]:
                paramsDict = dict(
                    dataset=data,
                    train_iters=200,
                    patience=100
                )
            elif type(model) in [CPGNN]:
                paramsDict = dict(
                    dataset=data,
                    train_iters=400,
                    patience=100
                )
            elif type(model) in [GraphSage]:
                paramsDict = dict(
                    feat_data=data.features,
                    labels=data.labels,
                    adj_matrix=data.adj,
                    train=data.idx_train,
                    val=data.idx_val,
                    train_iters=100,
                    lr=0.7
                )
            elif type(model) in [JKNet]:
                paramsDict = dict(
                    adj=data.adj,
                    features=data.features,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    idx_test=data.idx_test,
                    verbose=False
                )
            elif type(model) in [ProGNN]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    lr=0.01,
                    lr_adj=0.01,
                    epochs=200
                )
            elif type(model) in [GNNGuard]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    idx_test=data.idx_test,
                    train_iters=81, verbose=False
                )
            elif type(model) in [GAT]:
                paramsDict = dict(
                    dpr_data=data,
                    train_iters=1000,
                    verbose=False,
                    patience=100
                )
            elif type(model) in [GCNSVD]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    verbose=False
                )
            elif type(model) in [GPRGNN]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    verbose=False

                )
            elif type(model) in [FAGCN]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    verbose=False
                )
            elif type(model) in [APPNP]:
                paramsDict = dict(
                    features=data.features,
                    adj=data.adj,
                    labels=data.labels,
                    idx_train=data.idx_train,
                    idx_val=data.idx_val,
                    train_iters=200,
                    verbose=False

                )

            paramsDict.update(kwargs)
            print(f"<=== Training Model {name} \n {model} \n")
            start_ = datetime.now()
            model.fit(**paramsDict)
            print(f"Trained in {datetime.now() - start_} seconds.")
            print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
            model.eval()
            prediction_result = model.predict()
            print(f"<=== Complete training Model {name}")
            print(
                f"<=== Train Accuracy {self.check_correctness(prediction_result, data.idx_train).mean():.2%}")
            print(
                f"<=== Validation Accuracy {self.check_correctness(prediction_result, data.idx_val).mean():.2%}")
            print(
                f"<=== Test Accuracy {self.check_correctness(prediction_result, data.idx_test).mean():.2%} \n")
            if save_predictions:
                self.predictionDict[f"f:{name}@{data_name}"] = self.torch2Numpy(
                    prediction_result)
        return self

    fit_model = fit_models

    def save_model(self, model, target_path):
        if type(model) in [H2GCN, CPGNN]:
            checkpointPath = model.save_checkpoints(target_path)
            if checkpointPath:
                print(f"Checkpoint saved to {checkpointPath}")
            return checkpointPath

    def get_models(self, model_names):
        if type(model_names) is str:
            model_names = [model_names]
        return {name: self.modelDict[name] for name in model_names}

    @recordcalls
    def evaluate_models(self, model_names, target="idx_test"):
        raise NotImplementedError()
        return self

    def check_correctness(self, prediction: torch.Tensor, mask=None, data=None):
        if data is None:
            data = self.data
        pred_class = prediction.argmax(1).cpu().numpy()
        if mask is not None:
            return pred_class[mask] == data.labels[mask]
        else:
            return pred_class == data.labels

    def getData(self, name):
        if name == "clean":
            return deepcopy(self.data)
        elif name in self.perturbDict:
            if name.startswith("clean"):
                return deepcopy(self.data)
            perturbed_data = deepcopy(self.data)
            perturbed_data.adj = self.perturbDict[name]["adj"] + self.data.adj
            perturbed_data.features = self.perturbDict[name]["features"] + \
                self.data.features
            return perturbed_data
        else:
            raise KeyError(f"{name}")

    class JobExistsError(Exception):
        def __init__(self, jobID, *args, **kwargs):
            self.jobID = jobID
            super().__init__(*args, **kwargs)

    def create_signac_job(self, statePointDict, save_declared_model_keys=True, sp_include_init=True):
        project = signac.get_project()
        if sp_include_init:
            statePointDict.update(
                self.sessionTrace["__init__"]  # pylint: disable=no-member
            )
        statePointDict["sessionTrace"] = self.sessionTrace  # pylint: disable=no-member
        if self.use_runner:
            statePointDict["use_runner"] = True
        job = project.open_job(statePointDict).init()
        if job.doc.get("success", False):
            if not self.overwriteJob:
                raise self.JobExistsError(
                    job.id, f"ERROR: Job {job.id} has already existed and succeeded!")
            else:
                print(f"WARNING: overwriting existing job {job.id}")
        job.reset()
        self.signacJobHist.append(job.id)
        if save_declared_model_keys:
            job.doc.declared_models = {
                key: type(val).__name__ for key, val in self.modelDict.items()}
        return job

    def add_comment(self, comment):
        if self.signacJob:
            self.signacJob.doc.comment = comment
        return self

    def add_tag(self, tag):
        if self.signacJob:
            self.signacJob.doc.tag = tag
        return self

    def set_configs(self, **configs):
        for key, val in configs.items():
            print(f"Set self.{key} to {val}")
            assert hasattr(self, key)
            setattr(self, key, val)
        return self

    @staticmethod
    def torch2Numpy(tensor: torch.Tensor):
        return tensor.cpu().detach().numpy()

    @recordcalls
    def save_predictions(self, model_names, data_name, target="idx_test"):
        data = self.getData(data_name)
        for name, model in self.get_models(model_names).items():
            model.eval()
            model_output = model.predict()
        raise NotImplementedError()
        return self

    def exit(self):
        return

    def save_perturb(self, statePointDict, save_predictions=True, save_models=False, comment=None, tag=None):
        self.signacJob = self.create_signac_job(statePointDict)
        self.signacJobPerturb = self.signacJob
        with self.signacJob:
            # Store original graph and features
            with open("data.pkl", "wb") as f:
                pickle.dump(self.data, f)

            # Store perturbed graphs and features (as delta of the original)
            with open("perturbDict.pkl", "wb") as f:
                pickle.dump(self.perturbDict, f)
            self.signacJob.doc.perturbation_saved = True

            # Store predictions recorded for existing models
            if save_predictions and len(self.predictionDict) > 0:
                with self.signacJob.data:
                    self.signacJob.data.predictionDict = self.predictionDict
                self.signacJob.doc.prediction_saved = True

                with self.signacJob:
                    self.resultTable.to_csv("resultTable.csv")
                self.signacJob.doc.resultTable_saved = True

            # Save models
            if save_models:
                for model_name, model in self.modelDict.items():
                    self.save_model(
                        model,
                        Path(self.signacJob.workspace()) /
                        "saved_models" / model_name
                    )
                self.signacJob.doc.model_saved = True
            self.signacJob.doc.comment = comment
            if tag is not None:
                self.add_tag(tag)
            if not self.debug:
                self.signacJob.doc.success = True
        print(f"Saved perturbations to signac job {self.signacJob.id}")
        return self

    @classmethod
    def resume_from_job(cls, jobId, debug=False, check_success=True, use_runner=False):
        if debug == "vs":
            cls._debug()
        project = signac.get_project()
        job = project.open_job(id=jobId)
        if check_success:
            assert job.doc.get(
                "success", False), f"Tried to resume from unsuccessful job {jobId}"

        try:
            initParamsDict = job.sp.sessionTrace["__init__"]._to_base()
        except AttributeError: # Fall back to previous API
            initParamsDict = job.sp.sessionTrace["__init__"]._as_dict()

        initParamsDict["datasetName"] = None
        if debug is not None:
            initParamsDict["debug"] = True
        session = cls(**initParamsDict)  # type: AttackSession
        session.debug = debug
        session.signacJob = job
        if job.sp.type == "perturbation":
            session.signacJobPerturb = session.signacJob
        session.signacJobHist = [session.signacJob.id]
        with session.signacJob:
            # Load original graph and features
            session.datasetName = session.signacJob.sp.datasetName
            with open("data.pkl", "rb") as f:
                session.data = pickle.load(f)

            # Load perturbed graphs and features (as delta of the original)
            with open("perturbDict.pkl", "rb") as f:
                session.perturbDict = pickle.load(f)

            # Load predictions recorded for existing models
            if session.signacJob.doc.get("prediction_saved", False):
                with job.data.open(mode='r'):
                    session.predictionDict = {key: np.array(val) for key, val
                                              in session.signacJob.data.predictionDict.items()}

            if session.signacJob.doc.get("resultTable_saved", False):
                session.resultTable = pandas.read_csv("resultTable.csv")

        try:
            session.sessionTrace = session.signacJob.sp.sessionTrace._to_base()
        except AttributeError: # Fall back to previous API
            session.sessionTrace = session.signacJob.sp.sessionTrace._as_dict()
        session.modelDict = {
            key: Enum(val, "obj").obj for key, val
            in session.signacJob.doc.get("declared_models", dict()).items()
        }
        if use_runner:
            print(session.use_runner)
            session.use_runner = True
        return session
