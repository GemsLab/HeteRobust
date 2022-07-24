import contextlib
import pandas
import os
from pathlib import Path

@contextlib.contextmanager
def atomic_overwrite(filename):
    '''
    Adapted from https://stackoverflow.com/questions/42409707/pandas-to-csv-overwriting-prevent-data-loss
    '''
    temp = filename + '~'
    Path(temp).parent.mkdir(parents=True, exist_ok=True)
    with open(temp, "w") as f:
        yield f
    os.replace(temp, filename) # this will only happen if no exception was raised

def write_csv(df: pandas.DataFrame, filename: str):
    with atomic_overwrite(filename) as f:
        df.to_csv(f)

def list_value_format(input_list: list, value_mapping: dict):
    result_list = []
    for item in input_list:
        if type(item) is dict:
            result_list.append(dict_value_format(item, value_mapping))
        elif type(item) is list:
            result_list.append(list_value_format(item, value_mapping))
        elif type(item) is str:
            result_list.append(item.format(**value_mapping))
        elif type(item) in [int, float, bool] or item is None:
            result_list.append(item)
        else:
            raise ValueError()
    return result_list

def dict_value_format(input_dict: dict, value_mapping: dict):
    result_dict = dict()
    for key, value in input_dict.items():
        if type(value) is dict:
            result_dict[key] = dict_value_format(value, value_mapping)
        elif type(value) is list:
            result_dict[key] = list_value_format(value, value_mapping)
        elif type(value) is str:
            result_dict[key] = value.format(**value_mapping)
        elif type(value) in [int, float, bool] or value is None:
            result_dict[key] = value
        else:
            raise ValueError()
    return result_dict