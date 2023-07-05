#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0",
    description="Experiments with SuperNet and MNIST",
    author="Vasily Goncharov",
    url="https://github.com/basic-go-ahead/supernet-mnist",
    install_requires=[
        "tqdm",
        "pandas>=1.5.3",
        "matplotlib",
        "torch>=2.0.0",
        "torchvision>=0.15.1"
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_model = src.train:main",
            "eval_model = src.eval:main",
        ]
    },
)