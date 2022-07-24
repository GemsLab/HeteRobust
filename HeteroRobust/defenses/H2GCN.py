import subprocess
import sys
import os
from ..modules.communicate import PipeCommunicator
import tempfile
import torch
import threading
import time
import shutil
from pathlib import Path
import weakref
import signal
import traceback
from deeprobust.graph.data import Dataset


class SubprocessModel:

    def __init__(self, model_path, model_script, python_path=sys.executable, verbose=False):
        self.MODEL_PATH = model_path
        self.MODEL_SCRIPT = model_script
        self.CONDA_ENV = os.environ.get(
            f"__HR_{self.__class__.__name__}_CONDA_ENV__".upper(), ".")
        self.verbose = verbose
        self.PYTHON_PATH = python_path
        self.proc = None
        self.statusDaemon = None
        self.communicator = None
        self.set_python_path()
        self.tmp_dir = None # type: tempfile.TemporaryDirectory
        self.log_file = None

    def set_python_path(self):
        if self.CONDA_ENV and self.CONDA_ENV != ".":
            self.PYTHON_PATH = subprocess.run(["conda", "run", "-n", self.CONDA_ENV, "which", "python"],
                                              stdout=subprocess.PIPE, encoding="utf8", check=True).stdout.strip()
            print(f"Using {self.PYTHON_PATH}")
    
    def run_command_with_pipes(self, arg_list:list, stdout=None, stderr=None):
        # https://stackoverflow.com/questions/54060274/dynamic-communication-between-main-and-subprocess-in-python
        (r1, w1) = os.pipe2(0)  # for child -> parent writes
        (r2, w2) = os.pipe2(0)  # for parent -> child writes
        if not self.verbose:
            self.log_file = tempfile.NamedTemporaryFile(prefix="HR_SUBPROCESS_", suffix=".log", delete=False)
            print(f"Log file in {self.log_file.name}")
            stdout = self.log_file
            stderr = subprocess.STDOUT
        self.proc = subprocess.Popen([self.PYTHON_PATH, "-u", self.MODEL_SCRIPT] + arg_list + 
            ["--outpipe", str(w1), "--inpipe", str(r2)], 
            cwd=str(self.MODEL_PATH), pass_fds=[w1, r2],
            stdout=stdout, stderr=stderr)
        inpipe = os.fdopen(r1, 'rb')
        outpipe = os.fdopen(w2, 'wb')
        # self.communicator = PipeCommunicator(inpipe, outpipe)
        
        # The following implementation seems working...
        
        weakself = weakref.proxy(self)
        self.statusDaemon = threading.Thread(target=weakself.monitor_process, args=(weakself,), daemon=True)
        self.statusDaemon.start()
        self.communicator = PipeCommunicator(inpipe, outpipe, lambda: weakself.statusDaemon.join(timeout=5))
    
    @staticmethod
    def monitor_process(self):
        status = self.proc.poll()
        while status is None:
            time.sleep(1)
            status = self.proc.poll()
        exitCode = self.proc.wait()
        if exitCode != 0:
            try:
                raise RuntimeError(f"Subprocess {self.proc.args} (PID: {self.proc.pid}) created by {self} "
                             f"exited unexpectedly with return code {exitCode}. "
                             "Check the log at " + 
                             (self.log_file.name if self.log_file is not None else "terminal"))
            except RuntimeError as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                os.kill(os.getpid(), signal.SIGTERM)

    def end_process(self):
        pid = self.proc.pid
        self.communicator.signalExit()
        self.statusDaemon.join()
        # self.proc.wait()
        if self.tmp_dir:
            self.tmp_dir.cleanup()
            self.tmp_dir = None
        if self.proc.returncode == 0 and self.log_file is not None:
            os.remove(self.log_file.name)
        print(f"Exited subprocess {pid}...")

    @staticmethod
    def sp_mat_equal(sp_a, sp_b):
        if sp_a.shape != sp_b.shape:
            return False
        return (sp_a != sp_b).nnz == 0

    def __del__(self):
        if self.proc is not None:
            self.end_process()

class H2GCN(SubprocessModel):
    def __init__(
        self, dataset=None, network_setup="M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO",
        adj_nhood=["1", "2"], optimizer="adam", lr=0.01, adj_norm_type="sym",
        adj_svd_rank=0, l2_regularize_weight=5e-4, early_stopping=0,
        no_feature_normalize=False, random_seed=123, use_dense_adj=False, 
        verbose=False, debug=False, _initialize=False
    ):
        super().__init__("baselines/h2gcn", "interact.py", verbose=verbose)
        self.dataset = dataset
        if dataset is not None:
            self.dataset.__class__ = Dataset
        self.network_setup = network_setup
        self.adj_nhood = adj_nhood
        self.optimizer = optimizer
        self.adj_svd_rank = adj_svd_rank
        self.adj_norm_type = adj_norm_type
        self.lr = lr
        self.l2_regularize_weight = l2_regularize_weight
        self.early_stopping = early_stopping
        self.no_feature_normalize = no_feature_normalize
        self.random_seed = random_seed
        self.use_dense_adj = use_dense_adj
        self.debug = debug
        
        self._predictions = None
        
        self._initialized = False
        if _initialize:
            self.initialize()

    def initialize(self, dataset=None):
        # TODO: optimize initialization without creating new child process
        if self.proc is not None:
            self.end_process()
        self._predictions = None
        self.trained = False
        self.saved_paths = []
        args = [
            "H2GCN", "deeprobust", "--network_setup", self.network_setup, 
            "--early_stopping", str(self.early_stopping),
            "--random_seed", str(self.random_seed),
            "--optimizer", str(self.optimizer),
            "--lr", str(self.lr),
            "--l2_regularize_weight", str(self.l2_regularize_weight),
            "--adj_svd_rank", str(self.adj_svd_rank),
            "--adj_norm_type", str(self.adj_norm_type)
        ]
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="HR_H2GCN_")
        args += ["--checkpoint_dir", self.tmp_dir.name]
        args += ["--adj_nhood"] + self.adj_nhood
        if self.no_feature_normalize:
            args += ["--no_feature_normalize"]
        if self.use_dense_adj:
            args += ["--use_dense_adj"]
        if self.debug:
            args += ["--debug"]
        if dataset is None:
            if self.dataset is None:
                raise ValueError("Dataset must be provided!")
            dataset = self.dataset
        else:
            self.dataset = dataset
            self.dataset.__class__ = Dataset
        self.run_command_with_pipes(args)
        dataset.__class__ = Dataset
        self.communicator.sendObject(dataset)
        self.communicator.checkReady()
        self._initialized = True
    
    def fit(self, dataset=None, initialize=True, train_iters=200, patience=100,
            adj_norm_type=None, no_feature_normalize=None):
        if (adj_norm_type != self.adj_norm_type or
              no_feature_normalize != self.no_feature_normalize):
            if adj_norm_type is not None:
                self.adj_norm_type = adj_norm_type
            if no_feature_normalize is not None:
                self.no_feature_normalize = no_feature_normalize
            self.initialize(dataset)
        elif not self._initialized or (initialize and self.trained):
            self.initialize(dataset)
        elif (dataset is not None and dataset != self.dataset):
            self.initialize(dataset)
        else:
            self._predictions = None
            self.saved_paths = []
        confDict = dict(epochs=train_iters, early_stopping=patience)
        self.communicator.sendObject("set_params")
        self.communicator.sendObject(confDict)
        self.communicator.checkReady()

        self.communicator.sendObject("train")
        self.communicator.checkReady()
        self.trained = True

    def eval(self):
        pass

    def predict(self, features=None, adj=None):
        self.update_dataset(features, adj)
        if self._predictions is None:
            self.communicator.sendObject("predict")
            self._predictions = self.communicator.recvObject()
        return torch.log_softmax(torch.FloatTensor(self._predictions), dim=1)

    def update_dataset(self, features=None, adj=None, **kwargs):
        if features is not None:
            if not self.sp_mat_equal(features, self.dataset.features):
                kwargs["features"] = features
        
        if adj is not None:
            if not self.sp_mat_equal(adj, self.dataset.adj):
                kwargs["adj"] = adj

        for key, val in kwargs.items():
            assert hasattr(self.dataset, key)
            setattr(self.dataset, key, val)
        
        if len(kwargs) > 0:
            self._predictions = None
            self.communicator.sendObject("update_dataset")
            self.communicator.sendObject(kwargs)
            self.communicator.checkReady()

    def save_checkpoints(self, target_path):
        target_path = str(target_path)
        if target_path not in self.saved_paths:
            self.saved_paths.append(target_path)
            if Path(target_path).exists():
                shutil.rmtree(target_path)
            return shutil.copytree(self.tmp_dir.name, target_path)
    
    def enter_debug(self):
        self.communicator.sendObject("debug")
        self.communicator.checkReady()
    
class CPGNN(H2GCN):
    def __init__(
        self, dataset=None, network_setup='GGM64-VS-R-G-GMO-VS-E-BP2', 
        adj_nhood=['0','1','2'], optimizer='adam', lr=0.01, adj_norm_type='cheby', 
        adj_svd_rank=0, l2_regularize_weight=5e-4, early_stopping=0, 
        no_feature_normalize=False, random_seed=123, 
        verbose=False, debug=False, _initialize=False
    ):
        super().__init__(
            dataset, network_setup=network_setup, 
            adj_nhood=adj_nhood, optimizer=optimizer, 
            lr=lr, adj_norm_type=adj_norm_type, 
            adj_svd_rank=adj_svd_rank, l2_regularize_weight=l2_regularize_weight, 
            early_stopping=early_stopping, no_feature_normalize=no_feature_normalize, 
            random_seed=random_seed, verbose=verbose, debug=debug, 
            _initialize=False
        )

        if _initialize:
            self.initialize()

    def initialize(self, dataset=None):
        if self.proc is not None:
            self.end_process()
        self._predictions = None
        self.trained = False
        self.saved_paths = []
        args = [
            "CPGNN", "deeprobust", "--network_setup", self.network_setup, 
            "--early_stopping", str(self.early_stopping),
            "--random_seed", str(self.random_seed),
            "--optimizer", str(self.optimizer),
            "--lr", str(self.lr),
            "--l2_regularize_weight", str(self.l2_regularize_weight),
            "--adj_svd_rank", str(self.adj_svd_rank),
            "--adj_normalize", str(self.adj_norm_type)
        ]
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="HR_CPGNN_")
        args += ["--checkpoint_dir", self.tmp_dir.name]
        args += ["--adj_nhood"] + self.adj_nhood
        if self.no_feature_normalize:
            args += ["--no_feature_normalize"]
        if self.debug:
            args += ["--debug"]
        self.run_command_with_pipes(args)
        if dataset is None:
            dataset = self.dataset
        dataset.__class__ = Dataset
        self.communicator.sendObject(dataset)
        self.communicator.checkReady()
        self._initialized = True