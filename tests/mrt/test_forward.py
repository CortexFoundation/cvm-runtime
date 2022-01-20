from os import path

from mrt import dataset as ds
from mrt.V3.evaluate import inference_original_model, inference_quantized_model

def main():
    dataset_name = "coco"
    batch_size = 50
    input_shape = [batch_size,3,640,640]
    dataset = ds.DS_REG[dataset_name](input_shape)
    data_iter_func = dataset.iter_func()
    data, _ = data_iter_func()

    # test forward of original model
    symbol_file = path.expanduser("~/mrt_model/yolov5s.preprocess.unify.broadcastify.json")
    params_file = path.expanduser("~/mrt_model/yolov5s.preprocess.unify.broadcastify.params")
    outs = inference_original_model(
        symbol_file, params_file, data,
        batch_axis=0, device_type="gpu", device_ids=[0,1,2])
    print([o.shape for o in outs])

    # test forward of quantized model
    qsymbol_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.json")
    qparams_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.params")
    qext_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.ext")
    outs = inference_quantized_model(
        qsymbol_file, qparams_file, qext_file, data,
        batch_axis=0, split=False, device_type="gpu", device_ids=[0,1,2])
    print([o.shape for o in outs])


if __name__ == "__main__":
    main()
