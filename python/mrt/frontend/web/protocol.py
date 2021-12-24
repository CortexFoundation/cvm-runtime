import json

str2listofeval = lambda v: [eval(s) for s in v.split(',')]
str2eval = lambda v: eval(v)
str2listofstr = lambda v: [s.strip() for s in v.split(',')]

def str2bool(v):
    if v == "true":
        ret = True
    elif v == "false":
        ret = False
    else:
        raise RuntimeError("invalid v: {}".format(v))
    return ret

#  def str2attribute_deps(v):
    #  print(v)
    #  ret = json.loads(v)
    #  return ret

type_cast = {
    "COMMON": {
        "DEVICE_IDS": str2listofeval,
        "BATCH": str2eval,
        "RUN_EVALUATE": str2bool,
        "RUN_COMPILE": str2bool,
    },
    "PREPARE": {
        "DEVICE_IDS": str2listofeval,
        "INPUT_SHAPE": str2listofeval,
        "SPLIT_KEYS": str2listofstr,
    },
    "CALIBRATE": {
        "BATCH": str2eval,
        "NUM_CALIB": str2eval,
        "LAMBD": str2eval,
        "DEVICE_IDS": str2listofeval,
    },
    "QUANTIZE": {
        "RESTORE_NAMES": str2listofstr,
        "INPUT_PRECISION": str2eval,
        "OUTPUT_PRECISION": str2eval,
        "DEVICE_IDS": str2listofeval,
        "SOFTMAX_LAMBD": str2eval,
        "SHIFT_BITS": str2eval,
        # TODO ATTRIBUTE_DEPS, OSCALE_MAPS, THRESHOLDS
    },
    "EVALUATE": {
        "BATCH": str2eval,
        "DEVICE_IDS": str2listofeval,
        "ITER_NUM": str2eval,
    },
    "COMPILE": {
        "BATCH": str2eval,
        "DEVICE_IDS": str2listofeval,
    },
}
