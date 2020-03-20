
from ..base import _LIB, check_call

class SymbolBase(object):

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(_LIB().NNSymbolFree(self.handle))
