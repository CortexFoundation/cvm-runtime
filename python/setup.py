from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os

CURRENT_DIR = os.path.dirname(__file__)

def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, 'cvm/libinfo.py')
    libinfo = {'__file__': libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)
    version = libinfo['__VERSION__']
    if not os.getenv('CONDA_BUILD'):
        lib_path = libinfo['find_lib_path']()
        libs = [lib_path[0]]
        if libs[0].find("runtime") == -1:
            for name in lib_path[1:]:
                if name.find("runtime") != -1:
                    libs.append(name)
                    break
    else:
        libs = None
    return libs, version

LIB_LIST, VERSION = get_lib_path()

def config_cython():
    ret = []
    python_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(python_dir)
    path = "cvm/_cython"

    # library_dirs = [os.path.join(root_dir, "build")]
    # libraries = ["cvm"]
    library_dirs = None
    libraries = None

    for fn in os.listdir(path):
        if not fn.endswith(".pyx"):
            continue
        ret.append(Extension(
            "cvm._cy3.%s" % (fn[:-4]),
            [os.path.join(path, fn)],
            include_dirs=[os.path.join(root_dir, "include")],
            library_dirs=library_dirs,
            libraries=libraries,
            language="c"))
    return cythonize(ret, compiler_directives={"language_level": 3})

setup_kwargs = {}
curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
for i, path in enumerate(LIB_LIST):
    LIB_LIST[i] = os.path.relpath(path, curr_path)
setup_kwargs = {
    "include_package_data": True,
    "data_files": [('cvm', LIB_LIST)]
}

setup(name='cvm',
    version=VERSION,
    description="CVM: A Deterministic Inference Framework for Deep Learning",
    url="https://github.com/CortexFoundation/cvm-runtime.git",
    packages=find_packages(),
    author="CortexLabs Foundation",
    # ext_modules = config_cython(),
    **setup_kwargs)
