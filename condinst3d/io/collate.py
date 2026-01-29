from typing import Sequence, Set, Dict
from torch.utils.data._utils.collate import default_collate


def multi_instance_collate(
        batch: Sequence,
        collate_keys: Set[str],
        target_keys: Dict[str, Dict[str, str]]
):
    elem = batch[0]
    if isinstance(elem, list):
        data = [i for k in batch for i in k]
        keys = elem[0].keys()
    elif isinstance(elem, tuple):
        data = [k[0] for k in batch]
        keys = elem[0].keys()
    elif isinstance(elem, dict):
        data = batch
        keys = data[0].keys()
    else:
        raise NotImplementedError(f"Element of type {type(elem)} not supported!")

    targets = {x: {} for x in target_keys.keys()}
    ret = {}

    for key in keys:
        data_for_batch = [d[key] for d in data]
        if key in collate_keys:
            is_target = False
            for target_name, target_mapping in target_keys.items():
                if key in target_mapping.keys():
                    targets[target_name][target_mapping[key]] = data_for_batch
                    is_target = True
                    break
            if not is_target:
                ret[key] = default_collate(data_for_batch)
        else:
            ret[key] = data_for_batch

    # for targets: reformat dict of lists to list of dicts
    for target_name, target_data in targets.items():
        if len(target_data) > 0:
            keys = target_data.keys()
            values = zip(*target_data.values())
            # Create a list of dictionaries
            target_data = [dict(zip(keys, value)) for value in values]
            ret[target_name] = target_data

    if isinstance(elem, tuple):
        return ret, default_collate([k[1] for k in batch])
    return ret
