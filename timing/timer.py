import subprocess

argumentsDict = {
    "GCN": ("MultiLayerGCN", "--nhid 64 --nlayer 2" , "--train_iters 200 --patience 500"),
    "H2GCN": ("H2GCN", "", "--train_iters 200 --patience 0"),
    "GraphSAGE": ("H2GCN", "--adj_nhood \"['1']\" --network_setup I-T1-G-V-C1-M64-R-T2-G-V-C2-MO-R --adj_norm_type rw", "--train_iters 200 --patience 0"),
    "ProGNN": ("ProGNN", "--nhid 64", "--epochs 200"),
    "GNNGuard": ("GNNGuard", "--nhid 64 --base_model GCN_fixed", "--train_iters 200 --patience 500"),
    "GAT": ("GAT", "", "--train_iters 200 --patience 500"),
    "GCNSVD": ("GCNSVD", "--nhid 64 --svd_solver eye-svd --k 50", "--train_iters 200"),
    "H2GCN+SVD": ("H2GCN", "--adj_svd_rank 50", "--train_iters 200 --patience 0"),
    "GraphSAGE+SVD": ("H2GCN", "--adj_nhood \"['1']\" --network_setup I-T1-G-V-C1-M64-R-T2-G-V-C2-MO-R --adj_norm_type rw --adj_svd_rank 50", "--train_iters 200 --patience 0"),
    "CPGNN": ("CPGNN", "--network_setup M64-R-MO-E-BP2", "--train_iters 400 --patience 0"),
    "GPRGNN": ("GPRGNN", "--nhid 64 --alpha 0.9", "--train_iters 200"),
    "FAGCN": ("FAGCN", "--nhid 64 --eps 0.9 --dropout 0.5", "--train_iters 200")
}


for cur_model, val in argumentsDict.items():
    print(cur_model, val)
    try:
        exec_ = f"python -m HeteroRobust.run NettackSession --datasetName cora --random_seed 1709222957 --debug - add_model {val[0]} {val[1]} - fit_models m0 {val[2]} - exit > timing/logs/{cur_model}.log"
        subprocess.run(exec_, shell=True, check=True)
    except Exception as e:
        print(e, exec_)