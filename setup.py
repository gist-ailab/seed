#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

# Ensure the PyTorch version is compatible
torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

# Set the compiler
os.environ["CC"] = "gcc-9"
os.environ["CXX"] = "g++-9"

def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "mask_eee_rcnn", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime
        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str
        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "mask_eee_rcnn", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-g"]}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-g"
        ]

        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "mask_eee_rcnn._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

def get_model_zoo_configs() -> List[str]:
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "mask_eee_rcnn", "model_zoo", "configs"
    )
    if path.exists(destination):
        if path.islink(destination):
            os.unlink(destination)
        else:
            shutil.rmtree(destination)
    try:
        os.symlink(source_configs_dir, destination)
    except OSError:
        shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True)
    return config_paths

setup(
    name="mask_eee_rcnn",
    version=get_version(),
    author="FAIR",
    url="https://github.com/facebookresearch/mask_eee_rcnn",
    description="mask_eee_rcnn is FAIR's next-generation research platform for object detection and segmentation.",
    packages=find_packages(exclude=("configs", "tests")),
    package_data={"mask_eee_rcnn.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.6",
    install_requires=[
        "termcolor>=1.1",
        "Pillow==6.2.2",  # torchvision currently does not work with Pillow 7
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore",
        "future",  # used by caffe2
        "pydot",  # used to save caffe2 SVGs
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": ["flake8", "isort", "black==19.3b0", "flake8-bugbear", "flake8-comprehensions"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
