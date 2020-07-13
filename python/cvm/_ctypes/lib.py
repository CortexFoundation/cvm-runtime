import os
import ctypes

from ..libinfo import find_lib_path

# hook for doc generator
doc_generator = os.environ.get('DOC_GEN', None) == 'True'

if doc_generator:
    _LIB_PATH = None
    _LIB_NAME = None
    _LIB = None
else:
    _LIB_PATH = find_lib_path()
    _LIB_NAME = os.path.basename(_LIB_PATH[0])
    _LIB = ctypes.CDLL(_LIB_PATH[0], ctypes.RTLD_GLOBAL)
