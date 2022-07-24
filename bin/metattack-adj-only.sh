for ptb_ratio in 0.20
do
	for DATASET in "cora" "citeseer" "fb100" "snap-patents-downsampled"
	do
		for SEED in 1709222957 3635683305 442986796
		do
			EXP_NAME="metattack-${ptb_ratio}-$DATASET-adj-only"
			python -m HeteroRobust.run MetattackSession \
				--random_seed $SEED --datasetName $DATASET - \
				add_model GCN --dropout 0 --with_relu False --nhid 64 - \
				fit_models m0 - \
				run_perturb m0 --attack_budget "num_edges*$ptb_ratio" --attack_features False - \
				save_perturb --comment $EXP_NAME - \
				exit
		done
	done
done