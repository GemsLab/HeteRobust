import numpy as np
import torch
import fire
import builtins
import itertools
import pandas
import signac
import sys
import pickle

from tqdm import tqdm
from .modules.nettack import Nettack
from .selector import nettack_selector, random_selector
from copy import copy, deepcopy
from pathlib import Path
from enum import Enum
from .base import recordcalls, AttackSession


class NettackSession(AttackSession):
    @recordcalls
    def __init__(self, datasetName="citeseer",
                 direct=True, n_influencers=5,
                 attack_types=3, debug=False, random_seed=15, **kwargs):
        super().__init__(datasetName=datasetName, debug=debug, random_seed=random_seed, **kwargs)
        self.targetDict = dict()
        self.resultTable = pandas.DataFrame(
            columns=["attack_type", "defense_name", "perturb_name", "target_node", "run_ind", "acc"])

    @recordcalls
    def select_nodes(self, selector_name, model_names, **kwargs):
        selector = eval(selector_name)
        models = list(self.get_models(model_names).values())
        self.targetDict = selector(models, self.data, **kwargs)
        print(
            f"<=== {selector_name}: attack targets selected on {model_names} \n {self.targetDict} \n")
        return self

    @property
    def targets(self):
        return np.unique(list(itertools.chain(*self.targetDict.values())))

    @recordcalls
    def run_perturb_multi(self, surrogate_name,
                          adj_attack_budget="degrees[target_node] / 2",
                          feature_attack_budget="features_nnz[target_node] / 2",
                          name_prefix=None):
        if name_prefix is None:
            name_prefix = f"p_{surrogate_name}"
        degrees = self.data.adj.sum(1).A1 #pylint: disable=unused-variable
        features_nnz = self.data.features.sum(1).A1 #pylint: disable=unused-variable
        data = self.getData("clean")
        for target_node in tqdm(self.targets):
            # Create attack model
            # First, attack features only
            self.attack_model = Nettack(
                self.modelDict[surrogate_name], nnodes=self.data.adj.shape[0],
                attack_structure=False, attack_features=True,
                device=self.device
            )
            self.attack_model.to(self.device)
            n_perturbations = int(np.ceil(eval(feature_attack_budget)))
            self.attack_model.attack(
                data.features,
                data.adj,
                data.labels,
                target_node,
                n_perturbations,
                verbose=False
            )

            # Store perturbed dataset
            perturbed_data_name = f"{name_prefix}_{target_node}_feature_only"
            self.perturbDict[perturbed_data_name] = dict(
                adj=(self.attack_model.modified_adj - self.data.adj),
                features=(self.attack_model.modified_features -
                          self.data.features)
            )

            # Next, attack adj only
            data = self.getData(perturbed_data_name)

            self.attack_model = Nettack(
                self.modelDict[surrogate_name], nnodes=self.data.adj.shape[0],
                attack_structure=True, attack_features=False,
                device=self.device
            )
            self.attack_model.to(self.device)
            n_perturbations = int(np.ceil(eval(adj_attack_budget)))

            self.attack_model.attack(
                data.features,
                data.adj,
                data.labels,
                target_node,
                n_perturbations,
                verbose=False
            )

            # Store perturbed dataset
            perturbed_data_name = f"{name_prefix}_{target_node}"
            self.perturbDict[perturbed_data_name] = dict(
                adj=(self.attack_model.modified_adj - self.data.adj),
                features=(self.attack_model.modified_features -
                          self.data.features)
            )

        return self


    @recordcalls
    def run_perturb(self, surrogate_name, attack_budget="degrees[target_node]",
                    attack_structure=True, attack_features=True, name_prefix=None,
                    save_predictions=True, disable_wild=False):

        if name_prefix is None:
            name_prefix = f"p_{surrogate_name}"
        degrees = self.data.adj.sum(1).A1 #pylint: disable=unused-variable
        features_nnz = self.data.features.sum(1).A1 #pylint: disable=unused-variable
        data = self.getData("clean")

        num_targets = len(self.targets)
        misclassify_cnt = 0
        for target_node in tqdm(self.targets):
            # Create attack model
            self.attack_model = Nettack(
                self.modelDict[surrogate_name], nnodes=self.data.adj.shape[0],
                attack_structure=attack_structure, attack_features=attack_features,
                device=self.device
            )
            if disable_wild:
                self.attack_model.disabled_nodes = data.idx_wild

            self.attack_model.to(self.device)
            n_perturbations = int(eval(str(attack_budget)))
            self.attack_model.attack(
                data.features,
                data.adj,
                data.labels,
                target_node,
                n_perturbations,
                verbose=False
            )

            # Store perturbed dataset
            perturbed_data_name = f"{name_prefix}_{target_node}"
            self.perturbDict[perturbed_data_name] = dict(
                adj=(self.attack_model.modified_adj - self.data.adj),
                features=(self.attack_model.modified_features -
                          self.data.features)
            )

            # Run prediction on the perturbated data
            _perturbeData = self.getData(perturbed_data_name)
            _model = self.modelDict[surrogate_name]
            predictions = _model.predict(
                features=_perturbeData.features,
                adj=_perturbeData.adj
            )

            # Summarize the results
            acc = self.check_correctness(predictions, target_node)
            self.resultTable.loc[len(self.resultTable)] = pandas.Series(
                dict(defense_name=surrogate_name,
                     run_ind='0',
                     perturb_name=name_prefix,
                     target_node=target_node,
                     acc=acc, attack_type="evasion"
                     )
            )

            if acc == 0:
                misclassify_cnt += 1
            if save_predictions:
                self.predictionDict[
                    f"e:{surrogate_name}@{perturbed_data_name}"] = self.torch2Numpy(
                    predictions)

        print(
            f"Perturbation - misclassification rate {misclassify_cnt / num_targets}")

        return self

    run_poison = run_perturb

    def save_perturb(self, save_predictions=True, save_models=False, comment=None, tag=None):
        statePointDict = dict(
            type="perturbation",
            attackModel="nettack",
            targetNodes=self.targetDict
        )
        return super().save_perturb(statePointDict, save_predictions, save_models, comment, tag)

    save_poison = save_perturb

    @classmethod
    def resume_from_job(cls, jobId, debug=False, check_success=True, use_runner=False):
        session = super().resume_from_job(jobId, debug, check_success, use_runner)
        session.targetDict = session.signacJob.sp.targetNodes._as_dict()
        return session

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
                attackModel="nettack",
                targetNodes=self.targetDict,
                defenseModel=defense_model,
                modelParams=kwargs
        )
        signacJob = self.create_signac_job(statePointDict)
        num_targets = len(self.targets)
        for ind in range(num_runs):
            misclassify_cnt = 0
            for target_node in tqdm(self.targets):
                # Train defense model
                perturbed_data_name = f"{name_prefix}_{target_node}"
                model_name = f"p:{defense_model}@{ind}@{perturbed_data_name}"
                self.add_model(defense_model, name=model_name,
                               from_json=from_json, **kwargs)
                self.fit_models(
                    model_name, data_name=perturbed_data_name, **eval(train_params),
                    save_predictions=False
                ) # This function can also handle prediction saving

                predictions = self.modelDict[model_name].predict()
                acc = self.check_correctness(predictions, target_node)
                self.resultTable.loc[len(self.resultTable)] = pandas.Series(
                    dict(defense_name=defense_model, perturb_name=name_prefix,
                         target_node=target_node, run_ind=ind, acc=acc, attack_type="poison")
                )
                if acc == 0:
                    misclassify_cnt += 1
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
                    self.predictionDict[model_name] = self.torch2Numpy(
                        predictions)
            print(
                f"Round {ind} - misclassification rate {misclassify_cnt / num_targets}")

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
                     save_models=False, save_predictions=True):
        # Get the saved model
        # There should be only one model in the class in the evasion setting.
        _model = self.modelDict[defense_model]

        statePointDict = dict(
                type="test",
                attackPhase="evasion",
                attackModel="nettack",
                targetNodes=self.targetDict,
                defenseModel=type(_model).__name__,
                defenseModelInstance=defense_model
            )
        signacJob = self.create_signac_job(statePointDict)
        num_targets = len(self.targets)
        misclassify_cnt = 0

        for target_node in tqdm(self.targets):
            # Get the perturbated data
            perturbed_data_name = f"{name_prefix}_{target_node}"

            # Run prediction on the perturbated data
            _perturbeData = self.getData(perturbed_data_name)
            predictions = _model.predict(
                features=_perturbeData.features,
                adj=_perturbeData.adj
            )

            # Summarize the results
            acc = self.check_correctness(predictions, target_node)
            self.resultTable.loc[len(self.resultTable)] = pandas.Series(
                dict(defense_name=defense_model,
                     run_ind='0',
                     perturb_name=name_prefix,
                     target_node=target_node,
                     acc=acc, attack_type="evasion"
                     )
            )
            if acc == 0:
                misclassify_cnt += 1
            if save_predictions:
                self.predictionDict[f"e:{defense_model}@{perturbed_data_name}"] = self.torch2Numpy(
                    predictions)
        print(
            f"Evasion - misclassification rate {misclassify_cnt / num_targets}")

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
                    Path(self.signacJob.workspace()) / "saved_models" / defense_model
                )
                self.signacJob.doc.model_saved = True
            if not self.debug:
                self.signacJob.doc.success = True
        return self

    def print_result(self):
        print(self.resultTable)
        return self
