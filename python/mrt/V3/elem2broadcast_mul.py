import argparse

from mrt import tfm_ops as tops
from mrt.tfm_base import N
from mrt.sym_utils import 

@N.register_nm("broadcastify")
def broadcastify(sym, params):
    def callback(op, **kwargs):
        pass

class ElemwiseMul(tops.ElemwiseMul):
    def broadcastify(self, op, **kwargs):
        pass
