import itertools
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F


def createStandardBlock(in_channels: int, out_channels: int, kernel_size: int = 3, **kwargs: Dict[str, Any]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def createNASBlock(block_id: int, n_channels: int, n_varblocks: int) -> nn.Sequential:
    blocks = []
    for k in range(n_varblocks):
        b = createStandardBlock(n_channels, n_channels, padding=1, padding_mode="zeros")
        blocks.append((f"variable_block{block_id}{k}", b))
    return nn.Sequential(OrderedDict(blocks))


class SuperNet(nn.Module):
    def __init__(self, n_channels: int, architecture: Optional[Tuple[int, int]] = None, sampler = None):
        super(SuperNet, self).__init__()
        
        if architecture is None and sampler is None:
            raise ValueError("Необходимо задать конкретную архитектуру или алгоритм сэмплирования.")
        if architecture is not None and sampler is not None:
            raise ValueError("Нельзя фиксировать архитектуру и задавать алгоритм сэмплирования одновременно.")

        self.architecture = architecture
        self.sampler = sampler

        b = 32
        # Допустимые кол-ва подблоков в NAS-блоке.
        adm_nas = [1, 2, 3]
        
        self.start_block = createStandardBlock(n_channels, b)
        self.intermediate_block = createStandardBlock(b, b << 1, stride=2)
        self.gap_block = nn.AvgPool2d(7)
        self.linear = nn.Linear(64, 10, bias=False)
        
        if architecture is not None:
            if architecture[0] not in adm_nas or architecture[1] not in adm_nas:
                raise ValueError("Реализации NAS-блока могут содержать только 1, 2 или 3 изменяемых подблока.")
            nas_block1 = createNASBlock(block_id=1, n_channels=b, n_varblocks=architecture[0])
            nas_block2 = createNASBlock(block_id=2, n_channels=b<<1, n_varblocks=architecture[1])
            self.blocks = nn.Sequential(OrderedDict([
                ("start_block", self.start_block),
                ('NASBlock1', nas_block1),
                ("intermediate_block", self.intermediate_block),
                ('NASBlock2', nas_block2),
                ("global_average_pooling", self.gap_block)
            ]))
        else:
            self.nas_blocks = {}
            admissible_architectures = list(itertools.product(adm_nas, adm_nas))
            for arch in admissible_architectures:
                block_id = arch[0] * 10 + arch[1] * 100
                nas_block1 = createNASBlock(block_id=block_id + 1, n_channels=b, n_varblocks=arch[0])
                nas_block2 = createNASBlock(block_id=block_id + 2, n_channels=b<<1, n_varblocks=arch[1])
                setattr(self, f"nas_block{block_id + 1}", nas_block1)
                setattr(self, f"nas_block{block_id + 2}", nas_block2)
                self.nas_blocks[arch] = nas_block1, nas_block2
                
            self.sampler = self.sampler(admissible_architectures)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if self.architecture is not None:
            x = self.blocks(x)
        else:
            architecture = self.sampler.sample()
            x = self.start_block(x)
            nas_block1, nas_block2 = self.nas_blocks[architecture]
            x = nas_block1(x)
            x = self.intermediate_block(x)
            x = nas_block2(x)
            x = self.gap_block(x)
        x = x.contiguous().view(-1, 64)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)