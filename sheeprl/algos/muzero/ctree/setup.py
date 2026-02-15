import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

ext = Extension(
    "cytree",
    sources=["cytree.pyx"],
    extra_compile_args=["-O3"],
    include_dirs=[np.get_include()],
    language="c++"
)

setup(ext_modules=cythonize([ext]))
