import ctypes
import os
import numpy as np

from .. import libinfo
from ..base import check_call
from .lib import _LIB


class CVMContext(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int)]
    MASK2STR = {
        1 : 'cpu',
        2 : 'gpu',
        3 : 'formal',
    }
    STR2MASK = {
        'cpu' : 1,
        'gpu' : 2,
        'formal' : 3,
    }
    def __init__(self, device_type, device_id):
        super(CVMContext, self).__init__()
        self.device_type = device_type
        self.device_id = device_id

class CVMDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        0 : 'int',
        1 : 'uint',
        2 : 'float',
        4 : 'handle'
    }
    def __init__(self, type_str):
        super(CVMDataType, self).__init__()
        if isinstance(type_str, np.dtype):
            type_str = str(type_str)

        if type_str == "bool":
            self.bits = 1
            self.code = 1
            self.lanes = 1
            return

        arr = type_str.split("x")
        head = arr[0]
        self.lanes = int(arr[1]) if len(arr) > 1 else 1
        bits = 32


        if head.startswith("int"):
            self.code = 0
            head = head[3:]
        elif head.startswith("uint"):
            self.code = 1
            head = head[4:]
        elif head.startswith("float"):
            self.code = 2
            head = head[5:]
        elif head.startswith("handle"):
            self.code = 4
            bits = 64
            head = ""
        elif head.startswith("custom"):
            low, high = head.find('['), head.find(']')
            if not low or not high or low >= high:
                raise ValueError("Badly formatted custom type string %s" % type_str)
            type_name = head[low + 1:high]
            self.code = _api_internal._datatype_get_type_code(type_name)
            head = head[high+1:]
        else:
            raise ValueError("Do not know how to handle type %s" % type_str)
        bits = int(head) if head else bits
        self.bits = bits

    def __repr__(self):
        if self.bits == 1 and self.lanes == 1:
            return "bool"
        if self.code in CVMDataType.CODE2STR:
            type_name = CVMDataType.CODE2STR[self.code]
        else:
            type_name = "custom[%s]" % \
                        _api_internal._datatype_get_type_name(self.code)
        x = "%s%d" % (type_name, self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (self.bits == other.bits and
                self.code == other.code and
                self.lanes == other.lanes)

    def __ne__(self, other):
        return not self.__eq__(other)

class CVMArray(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p),
                ('ctx', CVMContext),
                ('ndim', ctypes.c_int),
                ("dtype", CVMDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]

class NDArrayBase(object):
    __slots__ = ["handle", "is_view"]
    def __init__(self, handle, is_view=False):
        self.handle = handle
        self.is_view = is_view

    def __del__(self):
        if not self.is_view and _LIB:
            check_call(_LIB.CVMArrayFree(self.handle))

    @property
    def _cvm_handle(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

CVMArrayHandle = ctypes.POINTER(CVMArray)

def _make_array(handle, is_view, is_container):
    handle = ctypes.cast(handle, CVMArrayHandle)
    return NDArray(handle, is_view)

def context(dev_type, dev_id=0):
    if isinstance(dev_type, str):
        dev_type = dev_type.split()[0]
        if dev_type not in CVMContext.STR2MASK:
            raise ValueError("Unknown device type %s" % dev_type)
        dev_type = CVMContext.STR2MASK[dev_type]
    return CVMContext(dev_type, dev_id)

def c_array(ctype, values):
    return (ctype * len(values))(*values)

def numpyasarray(np_data):
    """Return a CVMArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = CVMArray()
    shape = c_array(ctypes.c_int64, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = CVMDataType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape


def empty(shape, dtype="float32", ctx=context(1, 0)):
    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = CVMArrayHandle()
    dtype = CVMDataType(dtype)
    check_call(_LIB.CVMArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle, False, False)


CVMStreamHandle = ctypes.c_void_p

class NDArray(NDArrayBase):
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        t = CVMDataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.CVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<cvm.NDArray shape={0}, {1}>\n".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = CVMDataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.CVMArrayCopyToBytes(self.handle, data, nbytes))
        return np_arr

    def copyto(self, target):
        if isinstance(target, CVMContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.CVMArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target

def cpu(dev_id=0):
    return CVMContext(1, dev_id)


def gpu(dev_id=0):
    return CVMContext(2, dev_id)

def formal(dev_id=0):
    return CVMContext(3, dev_id)

def array(arr, ctx=cpu(0)):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : CVMContext, optional
        The device context to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)

class CVMByteArray(ctypes.Structure):
    _field_ = [('data', ctypes.c_void_p),
               ('size', ctypes.c_int64)]

def save(dict_data):
    data = []
    for k, v in dict_data.items():
        pk = ctypes.c_char_p(bytes(k, encoding='utf-8'))
        data.append(ctypes.cast(pk, ctypes.c_void_p))
        data.append(ctypes.cast(v.handle, ctypes.c_void_p))

    ret = CVMByteArray()
    arr = (ctypes.c_void_p * len(data))(*data)
    check_call(_LIB.CVMSaveParamsDict(ctypes.byref(arr), len(data), ctypes.byref(ret)))
    return ret

