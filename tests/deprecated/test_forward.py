from os import path

from mrt import dataset as ds
from mrt.V3.evaluate import inference_original_model, inference_quantized_model

def main():
    device_ids = [0,1,2]
    device_type = "gpu"

    dataset_name = "coco"
    # dataset_name = "voc"
    batch_size = 16 * len(device_ids)
    input_shape = [batch_size,3,640,640]
    # input_shape = [batch_size,3,416,416]
    dataset = ds.DS_REG[dataset_name](input_shape)
    data_iter_func = dataset.iter_func()
    data, _ = data_iter_func()

    # test forward of original model
    symbol_file = path.expanduser("~/mrt_model/yolov5s.preprocess.unify.broadcastify.json")
    params_file = path.expanduser("~/mrt_model/yolov5s.preprocess.unify.broadcastify.params")
    # symbol_file = path.expanduser("~/mrt_model/yolo3_darknet53_voc.json")
    # params_file = path.expanduser("~/mrt_model/yolo3_darknet53_voc.params")
    outs = inference_original_model(
        symbol_file, params_file, data,
        batch_axis=0, device_type=device_type, device_ids=device_ids)
    # print([o.shape for o in outs])
    print(outs[1])

    # test forward of quantized model
    qsymbol_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.json")
    qparams_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.params")
    qext_file = path.expanduser("~/mrt_model/yolov5s.mrt.quantize.ext")
    # qsymbol_file = path.expanduser("~/mrt_model/yolo3_darknet53_voc.all.quantize.json")
    # qparams_file = path.expanduser("~/mrt_model/yolo3_darknet53_voc.all.quantize.params")
    # qext_file = path.expanduser("~/mrt_model/yolo3_darknet53_voc.all.quantize.ext")
    outs = inference_quantized_model(
        qsymbol_file, qparams_file, qext_file, data,
        batch_axis=0, split=False, device_type=device_type, device_ids=device_ids)
    # print([o.shape for o in outs])
    print(outs[1])

if __name__ == "__main__":
    main()
