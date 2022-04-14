from typing import Dict, Tuple


def flatten_nested_dict(nested_dict: Dict, key: Tuple = ()) -> Dict:
    new_dict = {}
    for sub_key, value in nested_dict.items():
        new_key = (*key, sub_key)
        if isinstance(value, dict):
            new_dict.update(flatten_nested_dict(value, new_key))
        else:
            new_dict.update({new_key: value})
    return new_dict
