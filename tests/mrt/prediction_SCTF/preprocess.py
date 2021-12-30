from mrt import dataset as ds

from mxnet import ndarray as nd


@register_dataset("stdrandom")
class StdRandomDataset(ds.Dataset):
    def _load_data(self):
        def data_loader():
            N, I, C = self.ishape
            assert I == 1 and C == 3
            data, label = [], []
            while True:
                if len(data) < N:
                    x = nd.random.uniform(low=0.0,high=1.0,shape=(I,C))
                    y = nd.random.uniform(low=0.0,high=1.0,shape=(I))
                    data.append(x)
                    label.append(y)
                else:
                    yield nd.array(data), nd.array(label)
                    data, label = [], []
        self.data = data_loader()
