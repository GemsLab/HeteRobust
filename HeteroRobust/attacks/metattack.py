import numpy as np
import pickle
import pandas
import re
import signac

from pathlib import Path
from deeprobust.graph.global_attack.mettack import Metattack, MetaApprox
from deeprobust.graph.utils import preprocess
from scipy.sparse import csr_matrix
from .base import AttackSession, recordcalls

class MetattackSession(AttackSession):
    @recordcalls
    def __init__(self, datasetName="citeseer", variant="meta-self",
                 debug=False, random_seed=15, lr=0.01, momentum=0.9, with_bias=False):
        super().__init__(datasetName=datasetName, debug=debug, random_seed=random_seed)
        self.lr = lr
        self.momentum = momentum
        self.with_bias = with_bias
        self.resultTable = pandas.DataFrame(
            columns=["attack_type", "defense_name", "perturb_name", "run_ind",
                     "train_acc", "val_acc", "test_acc"])

        re_model = re.match(r"(a-)?((?:meta-)|(?:lambda-))(.+)", variant)
        if re_model:
            if re_model.group(1):
                self.use_MetaApprox = True
            else:
                self.use_MetaApprox = False

            if re_model.group(2) == "meta-":
                if re_model.group(3) == "self":
                    self.lambda_ = 0
                elif re_model.group(3) == "train":
                    self.lambda_ = 1
                elif re_model.group(3) == "both":
                    self.lambda_ = 0.5
            elif re_model.group(2) == "lambda-":
                self.lambda_ = float(re_model.group(3))
            else:
                raise ValueError(f"Unknown metattack variant {variant}.")

        else:
            raise ValueError(f"Unknown metattack variant {variant}.")

    @recordcalls
    def run_perturb(self, surrogate_name, attack_budget="num_edges*0.05", train_iters=100,
                    attack_structure=True, attack_features=True, name_prefix=None):
        if name_prefix is None:
            name_prefix = f"p_{surrogate_name}"
        data = self.getData("clean")
        # Performs the preprocessing as defined in metatack
        adj = data.adj
        features = data.features
        labels = data.labels
        adj, features, labels = preprocess(
            adj, features, labels, preprocess_adj=False)
        if self.use_MetaApprox:
            self.attack_model = MetaApprox(
                model=self.modelDict[surrogate_name],
                nnodes=adj.shape[0],
                feature_shape=features.shape,
                attack_structure=attack_structure, attack_features=attack_features,
                with_bias=self.with_bias, lambda_=self.lambda_, train_iters=train_iters, 
                lr=self.lr, device=self.device
            )
        else:
            self.attack_model = Metattack(
                model=self.modelDict[surrogate_name],
                nnodes=adj.shape[0],
                feature_shape=features.shape,
                attack_structure=attack_structure, attack_features=attack_features,
                with_bias=self.with_bias, lambda_=self.lambda_, train_iters=train_iters, 
                lr=self.lr, momentum=self.momentum, device=self.device
            )
        self.attack_model.to(self.device)

        num_edges = adj.sum() // 2 #pylint: disable=unused-variable
        if type(attack_budget) != int:
            attack_budget = int(eval(str(attack_budget)))

        self.attack_model.attack(
            ori_features=features,
            ori_adj=adj,
            labels=labels,
            idx_train=data.idx_train,
            idx_unlabeled=np.union1d(data.idx_val, data.idx_test),
            n_perturbations=attack_budget,
            ll_constraint=True,
            ll_cutoff=0.004
        )

        # Store perturbed dataset
        perturbed_data_name = f"{name_prefix}"
        if self.attack_model.modified_adj is not None:
            modified_adj = csr_matrix(self.attack_model.modified_adj.cpu().numpy())
        else:
            modified_adj = csr_matrix(data.adj)
        
        if self.attack_model.modified_features is not None:
            modified_features = csr_matrix(
                self.attack_model.modified_features.cpu().numpy())
        else:
            modified_features = csr_matrix(data.features)

        self.perturbDict[perturbed_data_name] = dict(
            adj=(modified_adj - data.adj),
            features=(modified_features - data.features)
        )
        return self

    run_poison = run_perturb

    def save_perturb(self, save_predictions=True, save_models=False, comment=None, tag=None):
        statePointDict = dict(
            type="perturbation",
            attackModel="metattack"
        )
        return super().save_perturb(statePointDict, save_predictions, save_models, comment, tag)
    
    save_poison = save_perturb

    @recordcalls
    def test_poison(self, name_prefix,
                    defense_model: str, from_json=None,
                    save_models=False, save_predictions=True,
                    num_runs=1, train_params="dict()",
                    **kwargs):

        self._noTrace = True
        statePointDict = dict(
            type="test",
            attackPhase="poison",
            attackModel="mettack",
            defenseModel=defense_model,
            modelParams=kwargs
        )
        signacJob = self.create_signac_job(statePointDict)
        for ind in range(num_runs):
            perturbed_data_name = f"{name_prefix}"
            model_name = f"p:{defense_model}@{ind}@{perturbed_data_name}"
            self.add_model(defense_model, name=model_name,
                           from_json=from_json, **kwargs)
            self.fit_models(
                model_name, data_name=perturbed_data_name, **eval(train_params),
                save_predictions=False)
            predictions = self.modelDict[model_name].predict()

            _perturbeData = self.getData(perturbed_data_name)
            train_acc = np.mean(self.check_correctness(
                predictions, _perturbeData.idx_train))
            val_acc = np.mean(self.check_correctness(
                predictions, _perturbeData.idx_val))
            test_acc = np.mean(self.check_correctness(
                predictions, _perturbeData.idx_test))
            self.resultTable.loc[len(self.resultTable)] = pandas.Series(dict(
                defense_name=defense_model,
                run_ind=ind,
                perturb_name=name_prefix,
                attack_type="poison",
                train_acc=train_acc,
                val_acc=val_acc,
                test_acc=test_acc
            ))
            print(f"<=== Poison test on {defense_model}: Train Accuracy {train_acc:.2%}; "
                  f"Val Accuracy {val_acc:.2%}; Test Accuracy {test_acc:.2%} \n")

            if save_models > 0:
                self.save_model(
                    self.modelDict[model_name],
                    Path(signacJob.workspace()) / "saved_models" / model_name
                )
                signacJob.doc.model_saved = True
            if save_models < 2:
                # save_models = False (0) / True (1) would not save the model in mem
                # Remove defense model
                self.modelDict.pop(model_name)
            if save_predictions:
                self.predictionDict[model_name] = self.torch2Numpy(predictions)

        if save_predictions:
            self.signacJob = signacJob
            if self.signacJobPerturb:
                self.signacJob.doc.perturbJob = self.signacJobPerturb.id
            with self.signacJob.data:
                self.signacJob.data.predictionDict = self.predictionDict
            self.signacJob.doc.prediction_saved = True
            print(f"Saved predictions to signac job {self.signacJob.id}")
            with self.signacJob:
                self.resultTable.to_csv("resultTable.csv")
            self.signacJob.doc.resultTable_saved = True
            if not self.debug:
                self.signacJob.doc.success = True
        self._noTrace = False
        return self

    @recordcalls
    def test_evasion(self, name_prefix, defense_model: str,
                     save_models=False, save_predictions=True,
                     save_clean_accuracy=True):
        # Get the saved model
        # There should be only one model in the class in the evasion setting.
        _model = self.modelDict[defense_model]

        statePointDict = dict(
            type="test",
            attackPhase="evasion",
            attackModel="mettack",
            defenseModel=type(_model).__name__,
            defenseModelInstance=defense_model
        )
        signacJob = self.create_signac_job(statePointDict)
        if save_clean_accuracy:
            predictions = _model.predict()
            _cleanData = self.data
            signacJob.doc.setdefault("performance", dict())
            signacJob.doc.performance.setdefault(f"f:{defense_model}@clean", dict())
            signacJob.doc.performance[f"f:{defense_model}@clean"]["train_acc"] = np.mean(
                self.check_correctness(predictions, _cleanData.idx_train))
            signacJob.doc.performance[f"f:{defense_model}@clean"]["val_acc"] = np.mean(
                self.check_correctness(predictions, _cleanData.idx_val))
            signacJob.doc.performance[f"f:{defense_model}@clean"]["test_acc"] = np.mean(
                self.check_correctness(predictions, _cleanData.idx_test))
        perturbed_data_name = f"{name_prefix}"
        _perturbeData = self.getData(perturbed_data_name)
        predictions = _model.predict(
            features=_perturbeData.features,
            adj=_perturbeData.adj)

        train_acc = np.mean(self.check_correctness(
            predictions, _perturbeData.idx_train))
        val_acc = np.mean(self.check_correctness(
            predictions, _perturbeData.idx_val))
        test_acc = np.mean(self.check_correctness(
            predictions, _perturbeData.idx_test))
        self.resultTable.loc[len(self.resultTable)] = pandas.Series(dict(
            defense_name=defense_model,
            run_ind='0',
            perturb_name=name_prefix,
            attack_type="evasion",
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc
        ))
        print(f"<=== Evasion test on {defense_model}: Train Accuracy {train_acc:.2%}; "
              f"Val Accuracy {val_acc:.2%}; Test Accuracy {test_acc:.2%} \n")
        if save_predictions:
            self.predictionDict[f"e:{defense_model}@{perturbed_data_name}"] = self.torch2Numpy(
                predictions)

        if save_predictions:
            self.signacJob = signacJob
            if self.signacJobPerturb:
                self.signacJob.doc.perturbJob = self.signacJobPerturb.id
            with self.signacJob.data:
                self.signacJob.data.predictionDict = self.predictionDict
            self.signacJob.doc.prediction_saved = True
            print(f"Saved predictions to signac job {self.signacJob.id}")
            with self.signacJob:
                self.resultTable.to_csv("resultTable.csv")
            self.signacJob.doc.resultTable_saved = True
            if save_models:
                self.save_model(
                    _model,
                    Path(self.signacJob.workspace()) /
                    "saved_models" / defense_model
                )
                self.signacJob.doc.model_saved = True
            if not self.debug:
                self.signacJob.doc.success = True
        return self

    def print_result(self):
        print(self.resultTable)
        return self