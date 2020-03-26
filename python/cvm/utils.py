import numpy as np

def argmax(out):
    return np.argmax(out)

def classification_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        print("Front 10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[:10]]))
        print("Last  10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[-10:]]))
        cat = argmax(tmp)
        print("Argmax output category: %d with possiblity %d" % \
              (cat, tmp[cat]))

def detection_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        for i in range(0, len(tmp), 6):
           if tmp[i] == -1:
               print ("Detect object number: %d" % (i // 6))
               break
           print (tmp[i:i+6])

def load_model(sym_path, prm_path):
    with open(sym_path, "r") as f:
        json_str = f.read()
    with open(prm_path, "rb") as f:
        param_bytes = f.read()
    return json_str.encode("utf-8"), param_bytes

def load_np_data(data_path):
    data = np.load(data_path)
    return data.tobytes()

