# How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications


Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, and Danai Koutra. 2022. *How does Heterophily Impact Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications*. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA, 20 pages. https://doi.org/10.1145/3534678.3539418

[[Paper + Full Appendix]](https://arxiv.org/pdf/2106.07767.pdf)

## Additional Details on Empirical Evaluation

In the [full appendix](https://arxiv.org/pdf/2106.07767.pdf), we include additional details on the setups and results of empirical evaluation, which includes
- Implementations and detailed hyperparameters for GNNs and randomized smoothing;
- Details on combining heterophilous design with explicit robustness-enhancing mechanisms;
- Additional results for evasion attacks (§5.2) and certifiable robustness (§5.3);
- Discussions on the comparison between certifiable and empirical robustness.


## Requirements
1. Install a conda virtual environment with requirements

    ```
    conda env create -f HeteroRobustEnv.yml
    ```
2. Activate the new environment `conda activate torch`. The virtual environment is named as `torch`, and you can change it manually by modifying the first line of `HeteroRobustEnv.yml`.

    ```
    conda activate torch
    ```

3. Verify that the new environment was installed correctly:

    ```
    (torch) conda env list
    ```
    
    Should you meet any problem, refer to the [official conda document](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

4. Additionally, you will need to manually install TensorFlow==2.2 to run H2GCN, CPGNN and GraphSAGE , and make sure that you are running all the code with a CUDA-enabled GPU.

## Attack Generation
* To generate attacks with Nettack used for benchmark study (§5.2), run the executable file in `./bin/nettack-adj-only.sh`:

    ```
    (torch) chmod +x ./bin/nettack-adj-only.sh
    (torch) ./bin/nettack-adj-only.sh
    ```
    You can modify the configurations in `./bin/nettack-adj-only.sh`.
    * Change the random seed `SEED`
    * Change the `DATASET` you want to attack from `[cora, citeseer, fb100, snap-patents-downsampled]`
    * Change the number of target nodes `--random_num <Number_OF_TARGET_NODES>`

* To generate attacks with Metattack, run this command:

    ```
    (torch) chmod +x ./bin/metattack-adj-only.sh
    (torch) ./bin/metattack-adj-only.sh
    ```
    You can modify the configurations in `./bin/metattack-adj-only.sh`.
    * Change the random seed `SEED`
    * Change the `DATASET` you want to attack from `[cora, citeseer, fb100, snap-patents-downsampled]`
    * Change the percentage of the edges to be perturbed `ptb_ratio`. Please enter `ptb_ratio` as a float between [0, 1].

* To generate attacks with Nettack which we studied in §5.1:

    ```
    python -m HeteroRobust.runner configs/perturbation/nettack-single-perturb.json -p 4
    ```

## Evaluation

To perform robustness evaluation of the GNN models: 
1. Modify the `config` files in `./configs/` folder. 
    
    For Nettack and Metattack, 
    * You will need to first generate the perturbations as instructed in last section;
    * Please make sure `PerturbJobFilters/doc/comment` meet with your `EXP_NAME` in the attack phase (`/bin/nettack-adj-only.sh`  or  `./bin/metattack-adj-only.sh`);
    * Refer to the table below for the detailed arguments for each GNN model;
    * For each model, you can switch between poison attack only `poison`, evasion attack only `evasion`, and both poison attack and evasion attack `poison+evasion`;
    * Tunable hyperparameters are listed in `Vars`;
    * For Nettack, modify `./configs/adj-only-nettack/nettack-adj-only.json`;
    * For Metattack, modify `./configs/adj-only-metattack/metattack-adj-only.json`.

    For Certifiable robustness, 
    * You do not need to generate perturbations first;
    * Sparse smoothing parameters are listed in `SessionConfig` and `TemplateVars`;
    * Datasets used are listed in `datasetName`;
    * Modify `./configs/adj-only-cert/sparse-smoothing-cert.json`.
    

2. Run the evaluation
    ```eval
    python -m HeteroRobust.runner configs/<CONFIG_TO_RUN> -p <NUMBER_OF_WORKERS>
    ```
    For example, to evaluate with Nettack with 4 workers, run the following code.
    ```
    python -m HeteroRobust.runner configs/adj-only-nettack/nettack-adj-only.json -p 4
    ```
    To evaluate the certifiable robustness with 4 workers, run the following code.
    ```
    python -m HeteroRobust.runner configs/adj-only-cert/sparse-smoothing-cert.json -p 4
    ```

3. The outputs will be stored in the `workspace` folder. And you can analyze the results using the jupyter notebooks provided in the `results` folder. Refer to the Results section for more details.


Below are the model arguments used in experiments of the paper.

| GNN       | model_name | argument           |
|-----------|------------|--------------------|
| H2GCN-SVD | `H2GCN`      | `--adj_svd_rank {k}` (select the best `{k}` per dataset) |
| GraphSAGE-SVD | `H2GCN` | `--adj_nhood ['1'] --network_setup I-T1-G-V-C1-M64-R-T2-G-V-C2-MO-R --adj_norm_type rw --adj_svd_rank {k}` (select the best `{k}` per dataset) |
| H2GCN-MGDC | `H2GCN` | `--network_setup M64-R-T1-GS-V-T2-GS-V-C1-C2-D0.5-MO --adj_norm_type gdc` |
| GraphSAGE-MGDC | `H2GCN` | `--adj_nhood ['1'] --network_setup I-T1-GS-V-C1-M64-R-T2-GS-V-C2-MO-R --adj_norm_type gdc` |
| H2GCN | `H2GCN` | (Empty) |
| GraphSAGE | `H2GCN` | `--adj_nhood ['1'] --network_setup I-T1-G-V-C1-M64-R-T2-G-V-C2-MO-R --adj_norm_type rw` |
| CPGNN | `CPGNN` | `--network_setup M64-R-MO-E-BP2` |
| GPR-GNN | `GPRGNN` | `--nhid 64 --alpha {gprgnn_alpha}` (select the best `{gprgnn_alpha}` overall) |
| FAGCN | `FAGCN` | `--nhid 64 --eps {fagcn_eps} --dropout 0.5` (select the best `{fagcn_eps}` overall)|
| APPNP | `APPNP` | `--alpha 0.9` |
| GNNGuard | `GNNGuard` | `--nhid 64 --base_model GCN_fixed` |
| ProGNN | `ProGNN` | `--nhid 64` |
| GCN-SVD | `GCNSVD` | `--nhid 64 --svd_solver eye-svd --k {k}` (select the best `{k}` per dataset) |
| GCN-MGDC | `H2GCN` | `--adj_nhood ['0,1'] --network_setup M64-GS-V-R-D0.5-MO-GS-V --adj_norm_type gdc` |
| GAT | `GAT` | (Empty) |
| GCN | `H2GCN` | `--adj_nhood ['0,1'] --network_setup M64-G-V-R-D0.5-MO-G-V` |
| MLP | `H2GCN` | `--network_setup M64-R-D0.5-MO` |



## Results

Run the jupyter notebook under `./results/` for analyzing the results.
* For Nettack, please run `./results/adj-only-nettack/nettack.ipynb`
* For Metattack, please run `./results/adj-only-metattack/metattack.ipynb`.
* For Certifiable robustness, please run `./results/adj-only-cert/cert.ipynb`

Refer to the Appendix of the paper for detailed results.

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.


## Citation

Please cite our work if you find it is helpful for your research:

```bibtex
@inproceedings{zhu2021graph,
  title={How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications},
  author={Zhu, Jiong and Jin, Junchen and Loveland, Donald and Schaub, Michael T and Koutra, Danai},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22)},
  year={2022}
}
```

If you make use of our code in your work, please also cite [DeepRobust](https://github.com/DSE-MSU/DeepRobust), upon which our code is built:
```bibtex
@article{li2020deeprobust,
  title={Deeprobust: A pytorch library for adversarial attacks and defenses},
  author={Li, Yaxin and Jin, Wei and Xu, Han and Tang, Jiliang},
  journal={arXiv preprint arXiv:2005.06149},
  year={2020}
}
```
