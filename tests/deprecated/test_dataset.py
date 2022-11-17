from mrt import dataset

print (dataset.DS_REG.keys())

def test_loader(name, input_shape):
    print ("TEST LOADER -----------",
            name, "----------------")
    ds = dataset.DS_REG[name](input_shape)
    data_iter_func = ds.iter_func()

    for i in range(5):
        data, label = data_iter_func()
        npy = data.asnumpy().flatten()
        size = npy.shape[0]

        print (data.shape, npy[0: 728].tolist())

#  test_loader("coco", (8, 3, 224, 224))
#  test_loader("voc", (8, 3, 224, 224))
#  test_loader("cifar10", (8, 3, 32, 32))
#  test_loader("quickdraw", (7, 1, 28, 28))
test_loader("mnist", (8, 1, 28, 28))
#  test_loader("trec", (38, 8))
