import os
import ctypes

from ..libinfo import find_lib_path

# hook for read the docs compilation
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

if read_the_docs_build:
    _LIB_PATH = None
    _LIB_NAME = None
    _LIB = None
else:
    _LIB_PATH = find_lib_path()
    _LIB_NAME = os.path.basename(_LIB_PATH[0])
    _LIB = ctypes.CDLL(_LIB_PATH[0], ctypes.RTLD_GLOBAL)
