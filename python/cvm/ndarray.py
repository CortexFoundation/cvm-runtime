import ctypes
from .common import context, cpu
from ._ctypes.ndarray import *
from ._ctypes.lib import _LIB
from ._base import numeric_types
from ._base import integer_types

import numpy as np
import sys

class NDArray(NDArrayBase):
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def ndim(self):
        "ndim of this array"
        return self.handle.contents.ndim

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

    def __setitem__(self, key, value):
        """Set ndarray value
        Previous: deep copy from value to self.
        shape of self and value must be the same.
        
        Present: set ``self[key] = value``
        Advanced indexing (like operator ``take``) is supported.

        Parameters
        ----------
        key: int, np.array or tuple of ints and np.array-es
            The length of a tuple must be no longer than self.ndim
            An array index is used for advanced indexing.
        value: int or an ndarray like object
            Shape of ``value`` must match the shape of selected indices.
        return: self
        """
        # if (not isinstance(in_slice, slice) or
        #         in_slice.start is not None
        #        or in_slice.stop is not None):
        #     raise ValueError('Array only support set from numpy array')
        # if isinstance(value, NDArrayBase):
        #    if value.handle is not self.handle:
        #         value.copyto(self)
        # elif isinstance(value, (np.ndarray, np.generic)):
        #     self.copyfrom(value)
        # else:
        #     raise TypeError('type %s not supported' % str(type(value)))
        key = indexing_key_expand_implicit_axes(key, self.shape)
        lk = len(key)
        if lk != self.ndim:
            raise RuntimeError(
                'too {} indices after normalization: expected `ndim` ({}) '
                'but got {}.'.format('few' if lk < self.ndim else 'many',
                self.ndim, lk)
            )
        use_advanced = use_advanced_indexing(key)   # some dim of the key is an array.
        if use_advanced:
            print('TODO: using advanced indexing setitem')
            pass
        else:
            print('using basic indexing setitem.')
            starts, ends, steps, sizes = expand_keys2_slice(key, self.shape)
            if sizes == self.shape and all(stp > 0 for stp in steps):   # overwrite all
                if isinstance(value, numeric_types):
                    if(isinstance(value, bool)):
                        self.full(int(value))
                    else:
                        self.full(value)
                else:
                    if not isinstance(value, self.__class__):
                        try:
                            value = array(value)
                        except:
                            raise TypeError('cannot assign value {} of type {} to '
                            'a slice of NDArray'.format(value, type(value)))
                    _LIB.CVMAssignAllND(self.handle, value.handle)
            elif isinstance(value, numeric_types):
                idxs = (ctypes.c_int * (4 * self.ndim))(*starts, *ends, *steps, *sizes)
                _LIB.CVMAssignSliceScalar(self.handle, idxs, ctypes.c_double(value))
            else:
                if not isinstance(value, self.__class__):
                    try:
                        value = array(value, self.context)
                    except:
                        raise TypeError('cannot assign value {} of type {} to '
                        'a slice of NDArray'.format(value, type(value)))
                idxs = (ctypes.c_int * (4 * self.ndim))(*starts, *ends, *steps, *sizes)
                assign_shape = tuple(filter(lambda x: x != 1, sizes))
                value = self.prepare_bcast_shape(value, assign_shape)
                _LIB.CVMAssignSliceND(self.handle, idxs, value.handle)

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
        res = "<cvm.NDArray shape={0} {1}>\t".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def full(self, value):
        _LIB.CVMAssignAllScalar(self.handle, ctypes.c_double(value))
        return self

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

    def as_runtime_input(self):
        return self.asnumpy().tobytes()

    def copyto(self, target):
        if isinstance(target, CVMContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.CVMArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target

    def prepare_bcast_shape(self, value, bcast_shape):
        """Return a broadcast ``NDArray`` of shape ``bcast_shape``
        with same context and dtype as ``self``.
        `value`: numeric types or array like.
        `bcast_shape`: a shape tuple.
        """
    
        # TODO: after implementing calling ops in python, use squeeze, broadcast
        # intead of using numpy.squeeze/broadcast

        # create value_nd from value. value_nd is an np.ndarray
        if isinstance(value, numeric_types):
            value_nd = np.full(bcast_shape, value, dtype=self.dtype)
        elif isinstance(value, NDArray):
            value_nd = value.asnumpy()
        else:
            try:
                value_nd = np.asarray(value, dtype=self.dtype)
            except:
                raise TypeError('{} does not support assignment with non-array-like '
                          'object {} of type {}'.format(self.__class__, value, type(value)))
        
        # data type conversion
        if not value_nd.dtype == self.dtype:
            value_nd = value_nd.astype(self.dtype)
        # squeeze leading length 1.
        if value_nd.ndim > len(bcast_shape):
            squeeze_axes = []
            for i in range(value_nd.nidm - len(bcast_shape)):
                if value_nd.shape[i] == 1:
                    squeeze_axes.append(i)
                else:
                    break
            if squeeze_axes:
                value_nd = value_nd.squeeze(squeeze_axes)
        if not value_nd.shape == bcast_shape:
            value_nd = np.broadcast_to(value_nd, bcast_shape)
        return array(value_nd, ctx=self.ctx)
        # final version of ``prepare_bcast_shape`` should be like this:
        # create value_nd from value. value_nd is an ndarray
        # if isinstance(value, numeric_types):
        #     value_nd = empty(bcast_shape, ctx=self.ctx, dtype=self.dtype).full(value)
        # elif isinstance(value, NDArray):
        #     if not value.ctx == self.ctx:
        #         raise ValueError('the target to assign should be of the same as '
        #         'the source ndarray in present version, but got target: {} '
        #         'and source: {}'.format(value.ctx, self.ctx))
        # else:
        #     try:
        #         value_nd = array(value, ctx=self.ctx, dtype=self.dtype)
        #     except:
        #         raise TypeError('{} does not support assignment with non-array-like '
        #                         'object {} of type {}'.format(self.__class__, value, type(value)))

        # # handle the cases like the following
        # # a = nd.zeros((3, 3)), b = nd.ones((1, 1, 1, 1, 3)), a[0] = b
        # # b cannot broadcast directly to a[0].shape unless its leading 1-size axes are trimmed
        # if value_nd.ndim > len(bcast_shape):
        #     squeeze_axes = []
        #     for i in range(value_nd.ndim - len(bcast_shape)):
        #         if value_nd.shape[i] == 1:
        #             squeeze_axes.append(i)
        #         else:
        #             break
        #     if squeeze_axes:
        #         value_nd = value_nd.squeeze(squeeze_axes)

        # if value_nd.shape != bcast_shape:
        #     if value_nd.size == 0:
        #         value_nd = value_nd.reshape(bcast_shape)
        #     else:
        #         value_nd = value_nd.broadcast_to(bcast_shape)
        # return value_nd



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

def _make_array(handle):
    handle = ctypes.cast(handle, CVMArrayHandle)
    return NDArray(handle, False)

def empty(shape, dtype="int32", ctx=cpu()):
    """ C wrapper method of NDArray generator.

        Notice: the allocated memory is supposed as empty, which means
        the memory is not formated and the real data created is random
        and unuseful.

        Returns
        =======
        nd_arr: :class:`cvm.ndarray.NDArray`
            An empty NDArray.
    """
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
    return _make_array(handle)

def array(arr, ctx=cpu()):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : :py:class:`cvm.CVMContext`
        The device context to create the array, CPU context by default.

    Returns
    -------
    ret : :py:class:`cvm.ndarray.NDArray`
        The created array
    """
    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr, np.int32)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)

def save_param_dict(dict_data):
    """ Transform the python :class:`cvm.ndarray.NDArray` handle into bytes.

        Returns
        =======
        seq: bytes
            The bytes binary of parameters dict.
    """
    data = []
    for k, v in dict_data.items():
        pk = ctypes.c_char_p(bytes(k, encoding='utf-8'))
        data.append(ctypes.cast(pk, ctypes.c_void_p))
        data.append(ctypes.cast(v.handle, ctypes.c_void_p))

    ret = CVMByteArray()
    arr = (ctypes.c_void_p * len(data))(*data)
    check_call(_LIB.CVMSaveParamsDict(ctypes.byref(arr), len(data), ctypes.byref(ret)))
    return ret.tobytes()

def load_param_dict(bytes_arr):
    ret = {}
    num = ctypes.c_int()
    names = ctypes.POINTER(ctypes.c_void_p)()
    values = ctypes.POINTER(ctypes.c_void_p)()

    check_call(_LIB.CVMLoadParamsDict(
            ctypes.c_char_p(bytes_arr),
            ctypes.c_int(len(bytes_arr)),
            ctypes.byref(num),
            ctypes.byref(names),
            ctypes.byref(values)))
    for i in range(num.value):
        name = str(ctypes.cast(names[i],
                ctypes.c_char_p).value, encoding="utf-8")
        value = NDArray(ctypes.cast(values[i],
                CVMArrayHandle), False)
        print (name, value)
        ret[name] = value
    return ret


def indexing_key_expand_implicit_axes(key, shape):
    """
    Make implicit axes explicit by adding ``slice(None)``
    and convert boolean array to integer array through `nonzero`.

    Examples
    --------
    >>> shape = (3, 4, 5)
    >>> indexing_key_expand_implicit_axes(np.s_[2, 1, 1], shape)
    (2, 1, 1)
    >>> indexing_key_expand_implicit_axes(np.s_[0], shape)
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[0, ...], shape)  # equivalent
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[:2, None, 0, ...], shape)
    (slice(None, 2, None), None, 0, slice(None, None, None))
    >>> bool_array = np.array([[True, False, True, False],
                               [False, True, False, True],
                               [True, False, True, False]], dtype=np.bool)
    >>> indexing_key_expand_implicit_axes(np.s_[bool_array, None, 0:2], shape)
    (array([0, 0, 1, 1, 2, 2], dtype=int64), array([0, 2, 1, 3, 0, 2], dtype=int64), None, slice(None, 2, None))

    Acknowledgement
    ---------------
    Thanks for insperiation from mxnet
    """
    if not isinstance(key, tuple):
        key = (key,)
    # We need to loop explicitly since tuple functions like `index()` or
    # `count()` use `==` internally, which doesn't play well with fancy
    # indexing.
    ell_idx = None
    nonell_key = []

    axis = 0
    for i, idx in enumerate(key):
        if idx is Ellipsis:
            if ell_idx is not None:
                raise IndexError(
                    'Cannot use more than one ellipsis (`...`) for indexing'
                )
            ell_idx = i
        else:
            if idx is None:
                continue
            elif isinstance(idx, np.ndarray) and idx.dtype == np.bool_:
                # Necessary size check before using `nonzero`
                check_boolean_array_dimension(shape, axis, idx.shape)

                # Add the arrays from the nonzero result to the index
                nonell_key.extend(idx.nonzero())
                axis += idx.ndim
            else:
                nonell_key.append(idx)
                axis += 1

    nonell_key = tuple(nonell_key)

    if ell_idx is None:
        # This handles the case of "too few" indices, e.g., `nd.zeros((2, 3))[0]`,
        # where the ellipsis is implicitly after the last entry.
        ell_idx = len(nonell_key)

    ell_ndim = len(shape) - len(nonell_key)
    expanded_key = (nonell_key[:ell_idx] +
                    (slice(None),) * ell_ndim +
                    nonell_key[ell_idx:])

    return expanded_key

def check_boolean_array_dimension(array_shape, axis, bool_shape):
    """
    Advanced boolean indexing is implemented through the use of `nonzero`.
    Size check is necessary to make sure that the boolean array
    has exactly as many dimensions as it is supposed to work with before the conversion
    """
    for i, val in enumerate(bool_shape):
        if array_shape[axis + i] != val:
            raise IndexError('boolean index did not match indexed array along axis {};'
                             ' size is {} but corresponding boolean size is {}'
                             .format(axis + i, array_shape[axis + i], val))

def use_advanced_indexing(key):
    for k in key:
        if isinstance(k, (NDArray, tuple, list, np.ndarray, range)):
            return True
        elif not (isinstance(k, slice) or isinstance(k, integer_types)):
            raise ValueError('NDArray doesnot support indexing with type {}'.format(k))
    return False

def _get_dim_size(start, stop, step):
    """Given start, stop, and step, calculate the number of elements
    of this slice.
    """
    assert step != 0
    if stop == start:
        return 0
    if step > 0:
        assert start < stop
        dim_size = (stop - start - 1) // step + 1
    else:
        assert stop < start
        dim_size = (start - stop - 1) // (-step) + 1
    return dim_size

def expand_keys2_slice(keys, shape):
    ret_key = []
    ret_size = []
    for i, idx in enumerate(keys):
        if isinstance(idx, slice):
            ranged_idx = idx.indices(shape[i])
            ret_key.append(ranged_idx)
            ret_size.append(_get_dim_size(*ranged_idx))
        elif isinstance(idx, integer_types):
            if idx < -shape[i] or idx >= shape[i]:
                raise IndexError('expecting index of the {}th dim in '
                    'range [-{}, {}), but got {}'.format(i,
                    shape[i], shape[i], idx))
            if idx < 0:
                idx += shape[i]
            ret_key.append((idx, idx + 1, 1))
            ret_size.append(1)
        else:
            raise TypeError('type of a component in an index'
                            ' can only be int, np.int or slice')
    ses = tuple(zip(*ret_key))
    return ses[0], ses[1], ses[2], tuple(ret_size)

