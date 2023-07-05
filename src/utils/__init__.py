import os
import random
import pandas as pd
import numpy as np
import torch


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)


def logged_metrics(logFilePath: str) -> pd.DataFrame:
    with open(logFilePath, "rt") as file:
        supernet_metrics= {}
        subnet_metrics = {}
        for line in file:
            parts = line.split("\t")
            if parts[0].lstrip("INFO:root:") == "subnet":
                subnet_metrics[parts[1]] = float(parts[2])
            else:
                supernet_metrics[parts[1]] = float(parts[2])
                
    return pd.DataFrame({
        "Подсеть": subnet_metrics.keys(),
        "Top-1 Acc на SuperNet": supernet_metrics.values(),
        "Top-1 Acc независимо": subnet_metrics.values()
    })