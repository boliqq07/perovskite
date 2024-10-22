import json

from monty.json import MontyDecoder


def sparse_pmgjson(json_file: str):
    dct = {json_file: json.load(json_file, cls=MontyDecoder)}
    return dct
