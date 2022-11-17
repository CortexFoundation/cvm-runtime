import torch.nn as nn

class FuseRule(object):
    """Template class of rules for model fusing."""
    def __init__(self):
        """
        State parameters here.
        """
        pass

    def add_module(self, m):
        """
        Args:
            m: item from .named_modules()
        Returns:
            None
        """
        assert type(m) is nn.Module
        raise NotImplementedError

    def names_lists(self):
        """
        Returns:
            names_lists: A list of name list
        """
        raise
