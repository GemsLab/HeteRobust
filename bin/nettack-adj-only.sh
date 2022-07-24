for DATASET in 'cora' 'citeseer' "fb100" 'snap-patents-downsampled'
do
	for SEED in 3635683305 442986796 1709222957
	do
		EXP_NAME="nettack-$DATASET-adj-only"
		python -m HeteroRobust.run NettackSession \
			--random_seed $SEED --datasetName $DATASET - \
			add_model GCN --dropout 0 --with_relu False --nhid 64 - \
			fit_models m0 - \
			select_nodes random_selector m0 --random_num 60 - \
			run_perturb m0 --attack_features False - \
			save_perturb --comment $EXP_NAME - \
			exit
	done
done
