import itertools
from collections import defaultdict
from typing import Callable, Optional, Tuple, TypedDict

import torch.nn as nn
import torch.nn.functional as F


admissible_NASBlock_sizes = [1, 2, 3]
"""Допустимое количество подблоков NAS-блока."""


admissible_architectures = list(itertools.product(admissible_NASBlock_sizes, admissible_NASBlock_sizes))
"""Допустимые типы архитектур подсетей."""


Architecture = Tuple[int, int]
"""
Тип архитектуры подсети, задаваемый парой, компоненты которой определяют
кол-во подблоков первого и второго NAS-блока соответственно.
"""


class ForwardOptions(TypedDict, total=False):
    """
    Опции вывода суперсети, позволяющие фиксировать архитектуру подсети и
    оценивать эту подсеть в рамках суперсети.
    """
    architecture: Architecture


class ConvOptions(TypedDict, total=False):
    padding: int
    padding_mode: str


def createStandardBlock(in_channels: int, out_channels: int, kernel_size: int = 3, **conv_options: ConvOptions) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, **conv_options),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SuperNet(nn.Module):
    """Представляет суперсеть или её подсеть в случае фиксации архитектуры."""
    def __init__(self, architecture: Optional[Architecture] = None, sampler: Optional[Callable] = None):
        """
        Превращает SuperNet в подсеть путём фиксации архитектуры `architecture` или
        задаёт алгоритм сэмплирования `Sampler` суперсети.

        ---
        Параметры:
            architecture : Optional[Architecture]
                конкретная архитектура подсети
            sampler : Optional[Sampler]
                алгоритм сэмплирования подсетей
        """
        super(SuperNet, self).__init__()
        
        if architecture is None and sampler is None:
            raise ValueError("Необходимо задать конкретную архитектуру или алгоритм сэмплирования.")
        if architecture is not None and sampler is not None:
            raise ValueError("Нельзя фиксировать архитектуру и задавать алгоритм сэмплирования одновременно.")

        self.architecture = architecture
        self.sampler = sampler

        input_channels = 1
        mid_channels1 = 32
        mid_channels2 = mid_channels1 << 1
        
        self.start_block = createStandardBlock(input_channels, mid_channels1)
        self.intermediate_block = createStandardBlock(mid_channels1, mid_channels2, stride=2)
        self.gap_block = nn.AvgPool2d(7)
        self.linear = nn.Linear(64, 10, bias=False)

        self.nas_blocks = defaultdict(list)
        if architecture is None:
            architecture = (max(admissible_NASBlock_sizes), ) * 2
        self._createNASBlock(1, mid_channels1, architecture[0])
        self._createNASBlock(2, mid_channels2, architecture[1])
        self._initialize_weights()

    def _createNASBlock(self, block_id: int, n_channels: int, n_varblocks: int):
        if n_varblocks not in admissible_NASBlock_sizes:
            raise ValueError("Реализации NAS-блока могут содержать только 1, 2 или 3 изменяемых подблока.")
        for k in range(n_varblocks):
            b = createStandardBlock(n_channels, n_channels, padding=1, padding_mode="zeros")
            nas_block_name = f"NAS_block{block_id}"
            self.nas_blocks[nas_block_name].append(b)
            setattr(self, f"{nas_block_name}_{k + 1}", b)
        
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, **forwardOptions: ForwardOptions):
        if self.architecture is not None:
            architecture = self.architecture
        else:
            architecture = forwardOptions["architecture"] if "architecture" in forwardOptions else self.sampler()

        x = self.start_block(x)
        for subblock in self.nas_blocks["NAS_block1"][:architecture[0]]:
            x = subblock(x)
        x = self.intermediate_block(x)
        for subblock in self.nas_blocks["NAS_block2"][:architecture[1]]:
            x = subblock(x)
        x = self.gap_block(x)
        x = x.contiguous().view(-1, 64)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)