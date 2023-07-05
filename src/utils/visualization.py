import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from ..models import admissible_architectures

def display_accuracy_dynamics(historyFolderPath: str):
    history_right = np.load(os.path.join(historyFolderPath, "right-hist.npy"))
    history_total = np.load(os.path.join(historyFolderPath, "total-hist.npy"))
    
    right_dynamics = np.cumsum(history_right, axis=1)
    total_dynamics = np.cumsum(history_total, axis=1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        accuracy_dynamics = np.nan_to_num(right_dynamics / total_dynamics)
        
    plt.figure(figsize=(10, 6))

    for k in range(len(admissible_architectures)):
        plt.plot(accuracy_dynamics[k], label=str(admissible_architectures[k]))

    plt.suptitle('Динамика точности подсетей при обучении суперсети')
    plt.legend()
    plt.xlabel('Число итераций обучения (обработанных батчей)', fontsize=14)
    plt.ylabel('Точность', fontsize=14)
    plt.show()