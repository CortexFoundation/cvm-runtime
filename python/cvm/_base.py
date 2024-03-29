# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# coding: utf-8
# pylint: disable=invalid-name, unused-import
""" ctypes library of nnvm and helper functions """
from __future__ import absolute_import

import os
import sys
import ctypes
import numpy as np

#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str
    numeric_types = (float, int, np.float32, np.int32)
    integer_types = (int, np.integer)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring
    integer_types = (int, long, np.integer)
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


nn_uint = ctypes.c_uint
OpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
GraphHandle = ctypes.c_void_p

# Global dict of str to symbol to initialize variables
_all_var_init = {}

def c_str(string):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)

def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type.

    Parameters
    ----------
    cptr : ctypes.POINTER(ctypes.c_char)
        pointer to the raw memory region
    length : int
        the length of the buffer

    Returns
    -------
    buffer : bytearray
        The raw byte memory buffer
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res

def ctypes2numpy_shared(cptr, shape):
    """Convert a ctypes pointer to a numpy array

    The result numpy array shares the memory with the pointer

    Parameters
    ----------
    cptr : ctypes.POINTER(mx_float)
        pointer to the memory region

    shape : tuple
        shape of target ndarray

    Returns
    -------
    out : numpy_array
        A numpy array : numpy array
    """
    if not isinstance(cptr, ctypes.POINTER(mx_float)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (mx_float * size).from_address(ctypes.addressof(cptr.contents))
    return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def ctypes2docstring(num_args, arg_names, arg_types, arg_descs, remove_dup=True):
    """Convert ctypes returned doc string information into parameters docstring.

    num_args : nn_uint
        Number of arguments.

    arg_names : ctypes.POINTER(ctypes.c_char_p)
        Argument names.

    arg_types : ctypes.POINTER(ctypes.c_char_p)
        Argument type information.

    arg_descs : ctypes.POINTER(ctypes.c_char_p)
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for i in range(num_args.value):
        key = py_str(arg_names[i])
        if key in param_keys and remove_dup:
            continue
        param_keys.add(key)
        type_info = py_str(arg_types[i])
        ret = '%s : %s' % (key, type_info)
        if arg_descs[i]:
            ret += '\n    ' + py_str(arg_descs[i])
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str

class Status:
    SUCCEED = 0
    ERROR_LOGIC = 1
    ERROR_RUNTIME = 2

def check_call(ret):
    if ret != Status.SUCCEED:
        raise Exception("API called with error code: %d" % ret)
