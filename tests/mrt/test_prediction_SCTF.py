from os import path
import sys

from mxnet import ndarray as nd
import numpy as np

from mrt.V3.utils import get_cfg_defaults, merge_cfg, override_cfg_args
from mrt.V3.execute import run
from mrt import dataset as ds


@ds.register_dataset("stdrandom")
class StdRandomDataset(ds.Dataset):
    def _load_data(self):
        def data_loader():
            N, I, C = self.ishape
            assert I == 1 and C == 3
            data, label = [], []
            while True:
                if len(data) < N:
                    x = np.random.uniform(low=0.0, high=1.0, size=(I,C))
                    y = np.random.uniform(low=0.0, high=1.0, size=(I))
                    data.append(x)
                    label.append(y)
                else:
                    batch_data, batch_label = nd.array(data), nd.array(label)
                    yield batch_data, batch_label
                    data, label = [], []
        self.data = data_loader()

if __name__ == "__main__":
    assert len(sys.argv) >= 1 and len(sys.argv)%2 == 1, \
        "invalid length: {} of sys.argv: {}".format(len(sys.argv), sys.argv)
    yaml_file = path.join(
        path.dirname(path.realpath(__file__)), "model_zoo", "prediction_SCTF.yaml")
    cfg = get_cfg_defaults()
    cfg = merge_cfg(yaml_file)
    cfg = override_cfg_args(cfg, sys.argv[1:])
    run(cfg)
