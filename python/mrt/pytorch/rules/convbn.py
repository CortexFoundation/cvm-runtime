from . import FuseRule

import re

_IDLE = 0
_CONV = 1

class ConvBNRuleByName(FuseRule):
    """Rule for searching plain BN after Convolution by default pytorch-generated name.
    Equivalence transformation with out loss in accuracy."""

    @staticmethod
    def getInfo(keyword, name):
        res = re.match(f"(.*){keyword}(\d*)", name)
        if res is None:
            return None, None, None
        return res.group(), res.group(1), res.group(2)

    def __init__(self):
        self._names_lists = list()
        self._idle()

    def _idle(self):
        self.cur_list = list()
        self.cur_prefix = None
        self.cur_suffix = None
        self.state = _IDLE

    def add_module(self, m):
        m_name = m[0]
        if self.state == _IDLE:
            check, prefix, suffix = self.getInfo("conv", m_name)
            if check is not None:
                self.cur_prefix = prefix
                self.cur_suffix = suffix
                self.cur_list = [m_name]
                self.state = _CONV
        elif self.state == _CONV:
            check, prefix, suffix = self.getInfo("bn", m_name)
            if check is None:
                self._idle()
            elif prefix != self.cur_prefix:
                self._idle()
            elif suffix == self.cur_suffix:
                self.cur_list.append(m_name)
                self._names_lists.append(self.cur_list)
                self._idle()
            else:
                self._idle()
        else:
            raise NotImplementedError

    def names_lists(self):
        return self._names_lists