from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

def config_cython():
    libraries = ["cvm_runtime_cpu"]
    library_dirs = [".."]

    ret = []
    path = "./"
    for fname in os.listdir(path):
        if not fname.endswith(".pyx"):
            continue

        ret.append(Extension(
            "cvm.core",
            [os.path.join(path, fname)],
            include_dirs=["../include"],
            libraries=libraries,
            library_dirs=library_dirs,
            language="c++",

            ))

    return cythonize(ret)

setup(name='CVM-Runtime Python Interface',
      ext_modules=config_cython())
