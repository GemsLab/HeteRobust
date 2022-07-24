import numpy as np
from deeprobust.graph.utils import classification_margin

def nettack_selector(target_models, dataset, high_num=10, low_num=10, random_num=20):
    # Process for target models
    target_gcn = target_models[0]
    target_gcn.eval()
    target_output = target_gcn.predict()
    margin_dict = {}

    for idx in dataset.idx_test:
        margin = classification_margin(target_output[idx], dataset.labels[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin

    # Check correctness for other models
    for model in target_models[1:]:
        model.eval()
        output = model.predict()
        drop_set = set()
        for idx in margin_dict:
            margin = classification_margin(output[idx], dataset.labels[idx])
            if margin < 0:
                drop_set.add(idx)
        for idx in drop_set:
            margin_dict.pop(idx)

    sorted_margins = sorted(margin_dict.items(),
                            key=lambda x: x[1], reverse=True)
    high = [int(x) for x, y in sorted_margins[: high_num]]
    
    if low_num == 0:
        low = []
        other = [x for x, y in sorted_margins[high_num:]]
    else:
        low = [int(x) for x, y in sorted_margins[-low_num:]]
        other = [x for x, y in sorted_margins[high_num: -low_num]]
    
    if random_num > 0:
        other = np.random.choice(other, random_num, replace=False).tolist()
    else:
        other = []
    
    return dict(
        high_confidence=high, low_confidence=low, random=other
    )

def random_selector(_, dataset, random_num=60):
    return dict(
        random=np.random.choice(dataset.idx_test, random_num, replace=False).tolist()
    )