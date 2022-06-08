class Model(object):
    def __init__(self):
        self._training = True
        self._children = []
        self._name = ''
        self._param = {}
        self._inputs = {}
        self._outputs = {}

    def parameters(self):
        return self._param
 
    def input_names(self):
        return list(self._inputs.keys())
    
    def output_names(self):
        return list(self._outputs.keys())

    def name(self):
        return self._name

    def named_parameters(self, recurse:bool=True):
        for param_name, param in self._param:
            yield (param_name, param)
        if recurse:
            for child_name, child in self._children:
                for param_name, param in child:
                    yield (child_name, param_name, param)

    def add_output(self, name):
        raise NotImplementedError

    def children(self):
        for child in self._children:
            yield child

    def named_children(self):
        for child in self._children:
            yield (child.name(), child)

    def train(self):
        self._training = True
        self._set_train()
        for child in self._children:
            child.train()

    def _set_train(self):
        raise NotImplementedError

    def eval(self):
        self._training = False
        self._set_eval()
        for child in self._children:
            child.eval()

    def _set_eval(self):
        raise NotImplementedError

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        raise NotImplementedError

    @staticmethod
    def _input_ready(input_names:list, data_dict:dict):
        for name in input_names:
            if name not in data_dict:
                return False
        return True
