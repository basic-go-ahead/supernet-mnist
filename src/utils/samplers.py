import os
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from typing import List, Tuple


Strategy = Tuple[int, int]
"""Стратегия в алгоритме сэмплирования."""


class Sampler(ABC):
    """Представляет базовую функциональность сэмплера."""
    def __init__(self, strategies: List[Strategy]):
        """Инициализирует сэмплер списком стратегий."""
        self._strategies = strategies
        self._current_call = 0
        self._current_index = None
        self._current_strategy = None
        #region История фидбека
        self._history_right = np.zeros((len(strategies), 30000), dtype=int)
        self._history_total = np.zeros((len(strategies), 30000), dtype=int)
        #endregion

    @property
    def strategies(self) -> List[Strategy]:
        """Возвращает список всех стратегий, участвующих в сэмплировании."""
        return self._strategies
    
    @property
    def currentStrategy(self):
        """Возвращает текущую стратегию (результат предыдущего вызова метода `sample`)."""
        return self._current_strategy
    
    @abstractmethod
    def sample(self) -> Strategy:
        """Сэмплирует и возвращает стратегию."""
        pass

    def __call__(self) -> Strategy:
        return self.sample()

    @abstractproperty
    def name(self) -> str:
        """Возвращает название алгоритма сэмплирования."""
        pass

    def feedback(self, n_right: int, n_total: int):
        """
        Принимает фидбек.

        ---
        Параметры:
            n_right : int
                число правильно классифицированных экземпляров
            n_total : int
                общее число обработанных экземпляров
        """
        self._history_right.itemset(self._current_index, self._current_call, n_right)
        self._history_total.itemset(self._current_index, self._current_call, n_total)
        self._current_call += 1

    def save(self, folderPath: str):
        """Сохраняет историю фидбека в указанную директорию."""
        np.save(os.path.join(folderPath, "right-hist"), self._history_right[:, :self._current_call])
        np.save(os.path.join(folderPath, "total-hist"), self._history_total[:, :self._current_call])


class UniformSampler(Sampler):
    """Представляет алгоритм равномерного сэмплирования."""
    def __init__(self, strategies: List[Strategy]):
        """Инициализирует сэмплер списком стратегий."""
        super(UniformSampler, self).__init__(strategies)
    
    def sample(self):
        self._current_index = k = np.random.randint(0, len(self._strategies))
        self._current_strategy = self._strategies[k]
        return self._current_strategy
    
    @property
    def name(self) -> str:
        return "uniform"


class EpsilonGreedySampler(Sampler):
    """Представляет сэмплер на основе алгоритма Epsilon-Greedy."""
    def __init__(self, strategies: List[Strategy], epsilon: float, n_init_rounds: int = 100):
        """
        Инициализирует сэмплер перечнем стратегий, параметром ε и числом раундом инициализации.

        ---
        Параметры:
            strategies : List[Strategy]
                перечень стратегий, участвующих в сэмплировании
            epsilon : float
                порог, определяющий использование равномерного сэмплирования
            n_init_rounds : int
                кол-во начальных раундом, в рамках которых алгоритм набирает статистику
                перед возможностью использования лидера
        """
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("Значение параметра `epsilon` должно быть в интервале (0, 1).")
        super(EpsilonGreedySampler, self).__init__(strategies)
        
        self._epsilon = epsilon
        self._n_init_rounds = n_init_rounds
        self._n_rights = defaultdict(int)
        self._n_total = defaultdict(int)
        
    def sample(self):
        p = np.random.rand()
        if p < self._epsilon or self._current_call < self._n_init_rounds:
            self._current_index = np.random.randint(0, len(self._strategies))
        else:
            self._current_index = np.argmax([self._n_rights[s] / self._n_total[s] if self._n_total[s] else  0 for s in self._strategies])
        self._current_strategy = self._strategies[self._current_index]
        return self._current_strategy
    
    def feedback(self, n_right: int, n_total: int):
        Sampler.feedback(self, n_right, n_total)
        self._n_rights[self._current_strategy] += n_right
        self._n_total[self._current_strategy] += n_total

    @property
    def name(self) -> str:
        return "epsilon-greedy"