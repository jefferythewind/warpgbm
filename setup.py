import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def get_cuda_arch():
    try:
        smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']
        )
        archs = list(set(line.strip() for line in smi.decode().split('\n') if line.strip()))
        return ';'.join(archs)
    except Exception:
        return "8.0"  # fallback for CI or non-GPU systems

# Set TORCH_CUDA_ARCH_LIST if not manually set
if CUDA_HOME is not None and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    arch_list = get_cuda_arch()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    print(f"[WarpGBM Setup] TORCH_CUDA_ARCH_LIST={arch_list}")
else:
    print("[WarpGBM Setup] Skipping arch detection (already set or no CUDA).")

def get_extensions():
    if CUDA_HOME is None:
        print("CUDA_HOME not found. Skipping CUDA extensions.")
        return []
    
    return [
        CUDAExtension(
            name="warpgbm.cuda.node_kernel",
            sources=[
                "warpgbm/cuda/histogram_kernel.cu",
                "warpgbm/cuda/best_split_kernel.cu",
                "warpgbm/cuda/node_kernel.cpp",
            ]
        )
    ]

setup(
    name="warpgbm",
    version="0.1.5",  # bump version to avoid reuse error
    description="Warp-speed GBDT with CUDA histogram acceleration",
    long_description="High-performance gradient boosted decision trees with GPU acceleration.",
    long_description_content_type="text/markdown",
    author="Pranshu Bahadur",
    license="GPL-3.0-only",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension} if CUDA_HOME else {},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
    include_package_data=True,
    zip_safe=False,
)
