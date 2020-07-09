from __future__ import absolute_import
import sys
import os

def find_lib_path():
    cvm_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(cvm_dir, "..", "..")

    dll_path = [cvm_dir, source_dir]
    dll_path.append(os.path.join(source_dir, "build"))

    dll_path = [os.path.realpath(x) for x in dll_path]

    lib_names = ["libcvm.so", "libcvm.dylib"]
    lib_dll_path = [os.path.join(p, ln) for ln in lib_names for p in dll_path]

    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        message = ('Cannot find the files.\n' +
                   'List of candidates:\n' +
                   str('\n'.join(lib_dll_path)))
        raise RuntimeError(message)
    return lib_found

__VERSION__ = "0.1.0"
