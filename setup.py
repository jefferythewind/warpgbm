import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig

# Try importing torch and checking CUDA support
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    torch_available = True
    has_cuda = torch.version.cuda is not None and CUDA_HOME is not None
except ImportError:
    torch_available = False
    has_cuda = False

def get_extensions():
    extensions = []
    if has_cuda:
        print(f"Building with CUDA (found at {CUDA_HOME})")
        extensions.append(
            CUDAExtension(
                name="warpgbm.cuda.node_kernel",
                sources=[
                    "warpgbm/cuda/histogram_kernel.cu",
                    "warpgbm/cuda/best_split_kernel.cu",
                    "warpgbm/cuda/binner.cu",
                    "warpgbm/cuda/predict.cu",
                    "warpgbm/cuda/node_kernel.cpp",
                ]
            )
        )
    else:
        if torch_available:
            print("PyTorch installed but CUDA not available. Skipping CUDA extensions.")
        else:
            print("PyTorch not found. Skipping CUDA extensions.")
    return extensions

# Read version
with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension} if torch_available else {},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
)
