"""
setup.py — builds the pmconv_ext Python extension.

Usage:
    pip install .                    # build + install
    pip install -e .                 # editable install (for development)
    python setup.py build_ext --inplace  # build in-place (for quick tests)
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="pmconv",
    version="0.2.0",
    description="Tensor-valued SplineCNN tensor ops for pmconv (C++ / PyTorch)",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=[
        CppExtension(
            name="pmconv_ext",
            sources=[
                "src/spline_conv.cpp",
                "src/pmconv_bindings.cpp",
            ],
            extra_compile_args=["-std=c++17", "-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
